package org.example

import ai.djl.Model
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.transformer.BertBlock
import ai.djl.nn.transformer.BertMaskedLanguageModelBlock
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.nio.file.Paths

/**
 * Program to infer masked words using the model trained with BertMlmTrainingParallel.kt.
 */
fun main() {
    val modelDir = Paths.get("build/model")
    val modelName = "bert-mlm"
    val vocabSize = 30522
    val embeddingSize = 768
    val maxSequenceLength = 128

    println("Starting model load: $modelName")

    // 1. Initialize Tokenizer and Model
    HuggingFaceTokenizer.newInstance("bert-base-uncased").use { tokenizer ->
        Model.newInstance(modelName).use { model ->

            // 2. Define Model Structure (Must match BertMlmTrainingParallel.kt)
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

            // Custom block definition (Same as training)
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
                }

            model.block = combinedBlock

            // 3. Load Parameters
            try {
                model.load(modelDir, modelName)
                println("Model loaded successfully.")
            } catch (e: Exception) {
                println("Error: Model file not found. Please run the training program first.")
                e.printStackTrace()
                return
            }

            // 4. Define Translator
            val translator =
                object : Translator<String, String> {
                    override fun processInput(
                        ctx: TranslatorContext,
                        input: String,
                    ): NDList {
                        val manager = ctx.ndManager
                        val encoding = tokenizer.encode(input)
                        var ids = encoding.ids

                        val maskTokenId = tokenizer.encode("[MASK]").ids[0]
                        val padId = tokenizer.encode("[PAD]").ids[0]

                        // Pad or truncate to maxSequenceLength
                        if (ids.size > maxSequenceLength) {
                            ids = ids.sliceArray(0 until maxSequenceLength)
                        } else if (ids.size < maxSequenceLength) {
                            val paddedIds = LongArray(maxSequenceLength) { padId }
                            System.arraycopy(ids, 0, paddedIds, 0, ids.size)
                            ids = paddedIds
                        }

                        // Find [MASK] indices
                        val maskedIndicesList = ArrayList<Long>()
                        for (i in ids.indices) {
                            if (ids[i] == maskTokenId) {
                                maskedIndicesList.add(i.toLong())
                            }
                        }

                        if (maskedIndicesList.isEmpty()) {
                            throw IllegalArgumentException("Input sentence must contain [MASK]. Found ids: ${ids.joinToString()}")
                        }

                        val maskedIndices = maskedIndicesList.toLongArray()
                        ctx.setAttachment("maskedIndices", maskedIndices)

                        // Create NDArrays
                        val inputIds = manager.create(ids).expandDims(0)
                        val typeIds = manager.zeros(Shape(1, maxSequenceLength.toLong()))
                        val masks = manager.ones(Shape(1, maxSequenceLength.toLong()))

                        // maskedIndices for model input: [batch, num_masks]
                        // Since we process one sentence at a time (batch=1), we expand dims
                        val maskedIndicesND = manager.create(maskedIndices).expandDims(0)

                        return NDList(inputIds, typeIds, masks, maskedIndicesND)
                    }

                    override fun processOutput(
                        ctx: TranslatorContext,
                        list: NDList,
                    ): String {
                        val logits = list[0]
                        val maskedIndices = ctx.getAttachment("maskedIndices") as LongArray

                        val sb = StringBuilder()

                        for (i in maskedIndices.indices) {
                            val tokenLogits = logits.get(i.toLong())
                            val predictedId = tokenLogits.argMax().getLong()
                            val word = tokenizer.decode(longArrayOf(predictedId))

                            if (sb.isNotEmpty()) sb.append(", ")
                            sb.append("[MASK] at ${maskedIndices[i]} -> $word")
                        }

                        return sb.toString()
                    }

                    override fun getBatchifier(): Batchifier? = null
                }

            // 5. Execute Inference
            model.newPredictor(translator).use { predictor ->
                val testSentences =
                    listOf(
                        "Deep learning is a subset of [MASK] learning.",
                        "The capital of France is [MASK].",
                        "I love to write [MASK] code.",
                        "The quick brown fox jumps over the lazy [MASK].",
                    )

                println("\n=== Inference Test Results ===")
                for (sentence in testSentences) {
                    try {
                        println("Input: $sentence")
                        val result = predictor.predict(sentence)
                        println("Prediction: $result")
                        println("---------------------------")
                    } catch (e: Exception) {
                        println("Inference Error: ${e.message}")
                        e.printStackTrace()
                    }
                }
            }
        }
    }
}
