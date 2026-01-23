package org.example

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.transformer.BertBlock
import ai.djl.nn.transformer.BertMaskedLanguageModelBlock
import ai.djl.training.DefaultTrainingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.min

/**
 * BERT MLM training system using BertMaskedLanguageModelBlock and BertMaskedLanguageModelLoss,
 * with GPU acceleration support.
 */
fun main() =
    runBlocking {
        val filePath = "train.txt"
        val batchSize = 10
        val maxSequenceLength = 128
        val numEpochs = 10

        // Check if GPU is available and determine the device
        val device = if (Engine.getInstance().gpuCount > 0) Device.gpu() else Device.cpu()
        println("Using device: $device")

        if (!Files.exists(Paths.get(filePath))) {
            System.err.println("$filePath not found.")
            return@runBlocking
        }

        // 1. Initialize Tokenizer, Manager, Model
        HuggingFaceTokenizer.newInstance("bert-base-uncased").use { tokenizer ->
            NDManager.newBaseManager(device).use { manager ->
                Model.newInstance("bert-mlm", device).use { model ->

                    val vocabSize = 30522
                    val embeddingSize = 768

                    // 2. Construct BertMaskedLanguageModelBlock
                    val bertBlock =
                        BertBlock
                            .builder()
                            .setTokenDictionarySize(vocabSize)
                            .optTransformerBlockCount(4)
                            .optAttentionHeadCount(8)
                            .optEmbeddingSize(embeddingSize)
                            .optMaxSequenceLength(maxSequenceLength)
                            .build()

                    val mlmBlock = BertMaskedLanguageModelBlock(bertBlock, Activation::gelu)

                    // Define custom block
                    val combinedBlock =
                        object : SequentialBlock() {
                            init {
                                add(bertBlock)
                                add(mlmBlock)
                            }

                            override fun forwardInternal(
                                ps: ai.djl.training.ParameterStore,
                                inputs: NDList,
                                training: Boolean,
                                params: ai.djl.util.PairList<String, Any>?,
                            ): NDList {
                                // inputs: [tokenIds, typeIds, masks, maskedIndices]
                                val tokenIds = inputs[0]
                                val typeIds = inputs[1]
                                val masks = inputs[2]
                                val maskedIndices = inputs[3]

                                // 1. Execute BertBlock (Encoder)
                                val bertInput = NDList(tokenIds, typeIds, masks)
                                val bertOutput = bertBlock.forward(ps, bertInput, training)
                                val sequenceOutput = bertOutput[0] // [batch, seq_len, embedding_size]

                                // 2. Get Embedding Table
                                val embeddingParam = bertBlock.tokenEmbedding.parameters[0].value
                                val embeddingTable = ps.getValue(embeddingParam, tokenIds.device, training)

                                // 3. Execute BertMaskedLanguageModelBlock (MLM Head)
                                val mlmInput = NDList(sequenceOutput, maskedIndices, embeddingTable)
                                val mlmOutput = mlmBlock.forward(ps, mlmInput, training)

                                return mlmOutput // [log_probs]
                            }

                            override fun initializeChildBlocks(
                                manager: NDManager,
                                dataType: ai.djl.ndarray.types.DataType,
                                vararg inputShapes: Shape,
                            ) {
                                // inputShapes: [tokenIds, typeIds, masks, maskedIndices]
                                val tokenShape = inputShapes[0]
                                val typeShape = inputShapes[1]
                                val maskShape = inputShapes[2]
                                val maskedIndicesShape = inputShapes[3]

                                // Initialize BertBlock
                                bertBlock.initialize(manager, dataType, tokenShape, typeShape, maskShape)

                                // Get output shapes of BertBlock
                                val bertOutputShapes = bertBlock.getOutputShapes(arrayOf(tokenShape, typeShape, maskShape))
                                val sequenceOutputShape = bertOutputShapes[0] // [batch, seq_len, embedding_size]

                                // Shape of Embedding Table (vocab_size, embedding_size)
                                val embeddingTableShape = Shape(vocabSize.toLong(), embeddingSize.toLong())

                                // Initialize BertMaskedLanguageModelBlock
                                // Expected input: [sequenceOutput, maskedIndices, embeddingTable]
                                mlmBlock.initialize(manager, dataType, sequenceOutputShape, maskedIndicesShape, embeddingTableShape)
                            }

                            override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
                                // Return final output shape (log_probs)
                                val batchSize = inputShapes[0].get(0)
                                val maskedCount = inputShapes[3].get(1)
                                return arrayOf(Shape(batchSize * maskedCount, vocabSize.toLong()))
                            }
                        }

                    model.block = combinedBlock

                    // 3. Training Configuration
                    // Use standard Loss.softmaxCrossEntropyLoss instead of BertMaskedLanguageModelLoss
                    // because our model already outputs predictions only for masked positions.
                    val loss =
                        ai.djl.training.loss.Loss
                            .softmaxCrossEntropyLoss("loss", 1.0f, -1, true, true)

                    val config =
                        DefaultTrainingConfig(loss)
                            .addTrainingListeners()

                    model.newTrainer(config).use { trainer ->
                        // Shapes for initialization
                        val batchShape = Shape(batchSize.toLong(), maxSequenceLength.toLong())
                        val maskedIndicesShape = Shape(batchSize.toLong(), 20)

                        trainer.initialize(batchShape, batchShape, batchShape, maskedIndicesShape)

                        // 4. Load Training Data
                        val allLines = Files.readAllLines(Paths.get(filePath))
                        val validLines = allLines.filter { it.trim().isNotEmpty() }.toMutableList()

                        val maskTokenId = tokenizer.encode("[MASK]").ids[0]
                        val padTokenId = tokenizer.encode("[PAD]").ids[0]

                        println("Start training: ${validLines.size} lines of data, $numEpochs epochs")

                        // 5. Epoch Loop
                        for (epoch in 1..numEpochs) {
                            println("--- Epoch $epoch / $numEpochs ---")
                            validLines.shuffle()

                            var epochLoss = 0f
                            var batchCount = 0

                            // Mini-batch processing
                            for (i in validLines.indices step batchSize) {
                                val currentBatchSize = min(batchSize, validLines.size - i)

                                // Preprocessing (Tokenization & Masking) executed in parallel with Coroutines
                                val batchData =
                                    coroutineScope {
                                        (0 until currentBatchSize)
                                            .map { b ->
                                                async(Dispatchers.Default) {
                                                    val text = validLines[i + b]
                                                    prepareData0(text, tokenizer, maxSequenceLength, vocabSize, maskTokenId, padTokenId)
                                                }
                                            }.awaitAll()
                                    }

                                // Model computation on GPU/CPU
                                trainer.newGradientCollector().use { gc ->
                                    val inputIds = Array(currentBatchSize) { batchData[it].first } // tokenIds
                                    val maskedIndices = Array(currentBatchSize) { batchData[it].second } // maskedIndices
                                    val labels = Array(currentBatchSize) { batchData[it].third } // maskedTokens (labels)

                                    val inputIndicesND = manager.create(inputIds)
                                    val maskedIndicesND = manager.create(maskedIndices)
                                    val labelsND = manager.create(labels)

                                    val typeIndicesND = manager.zeros(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))
                                    val maskIndicesND = manager.ones(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))

                                    // Forward pass: [tokenIds, typeIds, masks, maskedIndices]
                                    val outputs = trainer.forward(NDList(inputIndicesND, typeIndicesND, maskIndicesND, maskedIndicesND))

                                    // Loss calculation: [labels], [log_probs]
                                    // outputs is [batch * masked_count, vocab_size]
                                    // labelsND is [batch, masked_count], so flattening is required
                                    val flattenedLabels = labelsND.reshape(Shape(-1))

                                    // Filter to ignore label -1
                                    val validMask = flattenedLabels.neq(-1)
                                    val validLabels = flattenedLabels.get(validMask)
                                    val validOutputs = outputs.singletonOrThrow().get(validMask)

                                    if (validLabels.size() > 0) {
                                        val lossValue = trainer.loss.evaluate(NDList(validLabels), NDList(validOutputs))
                                        gc.backward(lossValue)
                                        trainer.step()
                                        epochLoss += lossValue.getFloat()
                                        batchCount++
                                    }
                                }
                            }
                            println("Epoch $epoch finished. Average Loss: ${if (batchCount > 0) epochLoss / batchCount else 0.0}")
                        }

                        // 6. Save Model
                        val modelDir = Paths.get("build/model")
                        if (!Files.exists(modelDir)) Files.createDirectories(modelDir)
                        model.save(modelDir, "bert-mlm")
                        println("Model saved: ${modelDir.toAbsolutePath()}")
                    }
                }
            }
        }
    }

/**
 * Tokenize text and apply standard BERT 15% masking (Executed on CPU)
 * Returns: Triple(inputIds, maskedIndices, maskedLabels)
 * maskedIndices: Indices of masked positions (position within sequence, not flattened index)
 * maskedLabels: Ground truth token IDs for masked positions
 */
fun prepareData0(
    text: String,
    tokenizer: HuggingFaceTokenizer,
    maxLen: Int,
    vocabSize: Int,
    maskId: Long,
    padId: Long,
): Triple<LongArray, LongArray, LongArray> {
    val random = ThreadLocalRandom.current()
    val encoding = tokenizer.encode(text)
    val ids = encoding.ids
    val inputIds = LongArray(maxLen)

    val maskedIndicesList = ArrayList<Long>()
    val maskedLabelsList = ArrayList<Long>()

    val limit = min(maxLen, ids.size)

    for (s in 0 until limit) {
        val originalId = ids[s]
        if (random.nextDouble() < 0.15) {
            // Target for prediction (label) with 15% probability
            maskedIndicesList.add(s.toLong())
            maskedLabelsList.add(originalId)

            val subR = random.nextDouble()
            inputIds[s] =
                when {
                    subR < 0.8 -> maskId

                    // 80% [MASK]
                    subR < 0.9 -> random.nextInt(vocabSize).toLong()

                    // 10% Random
                    else -> originalId // 10% Keep original
                }
        } else {
            inputIds[s] = originalId
        }
    }

    // Pad to max length
    if (limit < maxLen) {
        for (s in limit until maxLen) {
            inputIds[s] = padId
        }
    }

    // Pad maskedIndices and maskedLabels to fixed length or handle as variable length.
    // DJL's gatherFromIndices expects [batch, indices_per_seq], so
    // we need to align length within batch.
    // Here we simply set max to 20 and pad deficiency (e.g. with 0).
    // Note: If 0th token (CLS) is not masked, padding with 0 will reference CLS.
    // To minimize impact, padding parts should be ignored, but
    // BertMaskedLanguageModelBlock computes for all indices.
    // Strictly speaking, we need to align mask counts or use dynamic graph construction,
    // but here we simply fill with 0 and set label to -1 (ignore).

    val maxMasks = 20
    val maskedIndices = LongArray(maxMasks) { 0 }
    val maskedLabels = LongArray(maxMasks) { -1 } // Should be ignored in Loss calculation

    for (i in 0 until min(maxMasks, maskedIndicesList.size)) {
        maskedIndices[i] = maskedIndicesList[i]
        maskedLabels[i] = maskedLabelsList[i]
    }

    return Triple(inputIds, maskedIndices, maskedLabels)
}
