package org.example

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.LambdaBlock
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.transformer.BertBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.loss.Loss
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.ThreadLocalRandom
import java.util.function.Function
import kotlin.math.min

/**
 * BERT MLM training system combining GPU acceleration and coroutine parallel preprocessing
 */
fun main() =
    runBlocking {
        val filePath = "train.txt"
        val batchSize = 10
        val maxSequenceLength = 128
        val numEpochs = 10

        // Check if GPU is available (fallback to CPU if not)
        val device = if (Engine.getInstance().gpuCount > 0) Device.gpu() else Device.cpu()
        println("Using device: $device")

        if (!Files.exists(Paths.get(filePath))) {
            System.err.println("$filePath not found.")
            return@runBlocking
        }

        // 1. Initialize Tokenizer and Model (Specify Device)
        HuggingFaceTokenizer.newInstance("bert-base-uncased").use { tokenizer ->
            // Create a Manager associated with the specified device (GPU/CPU)
            NDManager.newBaseManager(device).use { manager ->
                Model.newInstance("bert-mlm", device).use { model ->

                    val vocabSize = 30522
                    val embeddingSize = 768

                    // 2. Model Structure
                    val bertEncoder =
                        BertBlock
                            .builder()
                            .setTokenDictionarySize(vocabSize)
                            .optTransformerBlockCount(4)
                            .optAttentionHeadCount(8)
                            .optEmbeddingSize(embeddingSize)
                            .optMaxSequenceLength(maxSequenceLength)
                            .build()

                    val mlmHead = Linear.builder().setUnits(vocabSize.toLong()).build()
                    model.block =
                        SequentialBlock().apply {
                            add(bertEncoder)
                            // BertBlock returns [sequence_output, pooled_output].
                            // We only need sequence_output for MLM.
                            // Use LambdaBlock to select the first output.
                            add(
                                LambdaBlock(
                                    Function { input: NDList -> NDList(input.head()) },
                                    "FirstOutputSelector",
                                ),
                            )
                            add(mlmHead)
                        }

                    // 3. Training Configuration
                    // Loss.softmaxCrossEntropyLoss arguments: (name, weight, classAxis, sparseLabel, fromLogit)
                    // We mistakenly passed -100 as classAxis. classAxis should be -1 (last dimension) for (N, C) input.
                    // The ignoreLabel functionality is not directly exposed in this factory method in older DJL versions or works differently.
                    // However, SoftmaxCrossEntropyLoss constructor might not support ignoreLabel directly in all versions or we are using the wrong factory.
                    // Let's check if we can set ignoreLabel differently or use a different constructor.
                    // Actually, looking at the error "Dimension out of range (expected to be in range of [-2, 1], but got -100)",
                    // it confirms that the 3rd argument is indeed 'classAxis', not 'ignoreLabel'.
                    // We should set classAxis to -1.
                    // To handle ignoreLabel=-100, we might need to instantiate SoftmaxCrossEntropyLoss directly if possible,
                    // or use a different approach (e.g. masking the loss manually).
                    // But wait, DJL's SoftmaxCrossEntropyLoss usually doesn't have an 'ignoreLabel' parameter in the constructor exposed via this static method.
                    // It seems we confused the arguments.

                    // Let's try to find a way to set ignoreLabel.
                    // If not available, we can manually mask the loss.
                    // For now, let's fix the classAxis to -1.
                    val loss = Loss.softmaxCrossEntropyLoss("loss", 1.0f, -1, true, true)

                    val config =
                        DefaultTrainingConfig(loss)
                            .addTrainingListeners()

                    model.newTrainer(config).use { trainer ->
                        // BertBlock expects multiple inputs (tokenIds, typeIds, masks).
                        // We initialize with 3 shapes.
                        val inputShape = Shape(batchSize.toLong(), maxSequenceLength.toLong())
                        trainer.initialize(inputShape, inputShape, inputShape)

                        val allLines = Files.readAllLines(Paths.get(filePath))
                        val validLines = allLines.filter { it.trim().isNotEmpty() }.toMutableList()

                        val maskTokenId = tokenizer.encode("[MASK]").ids[0]
                        val padTokenId = tokenizer.encode("[PAD]").ids[0]

                        println("Start training (Parallel data preparation + ${device.deviceType} computation)")

                        // 5. Training Loop
                        for (epoch in 1..numEpochs) {
                            println("--- Epoch $epoch ---")
                            validLines.shuffle()

                            var epochLoss = 0f
                            var batchCount = 0

                            for (i in validLines.indices step batchSize) {
                                val currentBatchSize = min(batchSize, validLines.size - i)

                                // Preprocessing is done with CPU parallel processing (Coroutines)
                                val batchData =
                                    coroutineScope {
                                        (0 until currentBatchSize)
                                            .map { b ->
                                                async(Dispatchers.Default) {
                                                    val text = validLines[i + b]
                                                    prepareData(text, tokenizer, maxSequenceLength, vocabSize, maskTokenId, padTokenId)
                                                }
                                            }.awaitAll()
                                    }

                                // Network computation is done on GPU
                                trainer.newGradientCollector().use { gc ->
                                    val inputIds = Array(currentBatchSize) { batchData[it].first }
                                    val labelIds = Array(currentBatchSize) { batchData[it].second }

                                    // Creating via manager(GPU) transfers data to GPU memory
                                    val inputIndices = manager.create(inputIds)
                                    val labelIndices = manager.create(labelIds)

                                    // Create dummy typeIds and masks
                                    val typeIndices = manager.zeros(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))
                                    val maskIndices = manager.ones(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))

                                    val outputs = trainer.forward(NDList(inputIndices, typeIndices, maskIndices))

                                    // Reshape outputs and labels to 2D and 1D respectively for SoftmaxCrossEntropyLoss
                                    // Output: (batch * seq_len, vocab_size)
                                    // Label: (batch * seq_len)
                                    val sequenceOutput = outputs.singletonOrThrow()
                                    val flattenedOutput = sequenceOutput.reshape(Shape(-1, vocabSize.toLong()))
                                    val flattenedLabel = labelIndices.reshape(Shape(-1))

                                    // Since we cannot easily set ignoreLabel in the Loss object with the current API usage,
                                    // we will manually mask the loss here.
                                    // 1. Calculate loss for all tokens.
                                    // 2. Create a mask where label != -100.
                                    // 3. Multiply loss by mask.
                                    // 4. Average over valid tokens.

                                    // However, trainer.loss.evaluate returns a scalar loss (usually averaged).
                                    // If we want to mask, we need the element-wise loss.
                                    // But SoftmaxCrossEntropyLoss in DJL might not return element-wise loss easily via 'evaluate'.
                                    // A workaround is to replace -100 labels with a valid label (e.g. 0) temporarily,
                                    // calculate loss, and then zero out the loss for those positions.
                                    // But 'evaluate' aggregates.

                                    // Alternative: Use a custom Loss or check if we can pass ignore_index to the backend.
                                    // DJL's SoftmaxCrossEntropyLoss implementation in Java does:
                                    // return NDArray.logSoftmax(classAxis).neg().mul(target).sum(classAxis).mean(); (simplified)
                                    // It doesn't seem to support ignoreLabel natively in the Java wrapper logic if it's not passed to backend.

                                    // Let's try to filter the data BEFORE passing to loss.
                                    // We can select only the indices where label != -100.
                                    // This is dynamic shape, which might be slow or tricky, but correct.

                                    val validMask = flattenedLabel.neq(-100) // 1 for valid, 0 for ignore
                                    // We need to keep only valid entries.
                                    // booleanMask is supported in DJL.
                                    val validLabel = flattenedLabel.get(validMask)
                                    val validOutput = flattenedOutput.get(validMask)

                                    // If no valid tokens in batch (rare but possible), skip
                                    if (validLabel.size() > 0) {
                                        val lossValue = trainer.loss.evaluate(NDList(validLabel), NDList(validOutput))
                                        gc.backward(lossValue)
                                        trainer.step()
                                        epochLoss += lossValue.getFloat()
                                        batchCount++
                                    } else {
                                        // Skip step
                                    }
                                }
                            }
                            println("Epoch $epoch finished. Average Loss: ${if (batchCount > 0) epochLoss / batchCount else 0.0}")
                        }

                        val modelDir = Paths.get("build/model")
                        if (!Files.exists(modelDir)) Files.createDirectories(modelDir)
                        model.save(modelDir, "bert-mlm")
                        println("Model saved.")
                    }
                }
            }
        }
    }

/**
 * Tokenization and masking (Executed on CPU)
 */
fun prepareData(
    text: String,
    tokenizer: HuggingFaceTokenizer,
    maxLen: Int,
    vocabSize: Int,
    maskId: Long,
    padId: Long,
): Pair<LongArray, LongArray> {
    val random = ThreadLocalRandom.current()
    val encoding = tokenizer.encode(text)
    val ids = encoding.ids
    val inputIds = LongArray(maxLen)
    val labelIds = LongArray(maxLen)

    val limit = min(maxLen, ids.size)

    for (s in 0 until limit) {
        val originalId = ids[s]
        if (random.nextDouble() < 0.15) {
            labelIds[s] = originalId
            val subR = random.nextDouble()
            inputIds[s] =
                when {
                    subR < 0.8 -> maskId
                    subR < 0.9 -> random.nextInt(vocabSize).toLong()
                    else -> originalId
                }
        } else {
            inputIds[s] = originalId
            labelIds[s] = -100 // Ignore label
        }
    }

    if (limit < maxLen) {
        for (s in limit until maxLen) {
            inputIds[s] = padId
            labelIds[s] = -100 // Ignore label
        }
    }

    return Pair(inputIds, labelIds)
}
