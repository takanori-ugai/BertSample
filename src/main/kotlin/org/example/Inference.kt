package org.example

import ai.djl.Model
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.nn.LambdaBlock
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.transformer.BertBlock
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.nio.file.Paths
import java.util.function.Function

/**
 * Test program to load a trained BERT model and infer words for [MASK] parts.
 */
fun main() {
    val modelDir = Paths.get("build/model")
    val modelName = "bert-mlm"
    val vocabSize = 30522
    val embeddingSize = 768
    val maxSequenceLength = 128
    val topK = 5

    println("Starting model load: $modelName")

    // 1. Initialize Tokenizer and Model
    HuggingFaceTokenizer.newInstance("bert-base-uncased").use { tokenizer ->
        Model.newInstance(modelName).use { model ->

            // 2. Define Model Structure (Must match BertMLM.kt used during training)
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

            // 3. Load Parameters
            try {
                model.load(modelDir, modelName)
                println("Model loaded successfully.")
            } catch (e: Exception) {
                println("Error: Model file not found. Please run the training program first.")
                return
            }

            // 4. Define Translator (Text -> Tensor -> Prediction Result)
            val translator =
                object : Translator<String, String> {
                    override fun processInput(
                        ctx: TranslatorContext,
                        input: String,
                    ): NDList {
                        // Add padding to match input length to maxSequenceLength
                        val encoding = tokenizer.encode(input)
                        var ids = encoding.ids
                        val validLength = minOf(ids.size, maxSequenceLength)

                        // Pad or truncate
                        if (ids.size > maxSequenceLength) {
                            ids = ids.sliceArray(0 until maxSequenceLength)
                        } else if (ids.size < maxSequenceLength) {
                            val padId = tokenizer.encode("[PAD]").ids[0]
                            val paddedIds = LongArray(maxSequenceLength) { padId }
                            System.arraycopy(ids, 0, paddedIds, 0, ids.size)
                            ids = paddedIds
                        }

                        ctx.setAttachment("ids", ids)

                        val manager = ctx.ndManager
                        val inputIds = manager.create(ids).expandDims(0)

                        // BertBlock expects 3 inputs: tokenIds, typeIds, masks
                        // We need to create dummy typeIds and masks
                        val seqLen = ids.size.toLong()
                        val typeIds =
                            manager.zeros(
                                ai.djl.ndarray.types
                                    .Shape(1, seqLen),
                            )
                        // Mark real tokens as valid and ignore the padding positions.
                        val attentionMask =
                            FloatArray(maxSequenceLength) { index ->
                                if (index < validLength) 1f else 0f
                            }
                        val masks = manager.create(attentionMask).expandDims(0)

                        return NDList(inputIds, typeIds, masks)
                    }

                    override fun processOutput(
                        ctx: TranslatorContext,
                        list: NDList,
                    ): String {
                        // Get the first batch from model output [batch, seq_len, vocab_size]
                        val logits = list[0][0] // [seq_len, vocab_size]
                        val ids = ctx.getAttachment("ids") as LongArray

                        // Identify the index of the [MASK] token.
                        val maskTokenId = tokenizer.encode("[MASK]").ids[0]
                        val maskIndex = ids.indexOf(maskTokenId)
                        if (maskIndex == -1) return "[MASK] not found."

                        // Return the top-k token IDs ranked by their logits.
                        val maskLogits = logits.get(NDIndex(maskIndex.toLong()))
                        val logitsArray = maskLogits.toFloatArray()
                        val topKIds =
                            logitsArray.indices
                                .sortedByDescending { logitsArray[it] }
                                .take(topK)

                        return topKIds.joinToString(", ") { tokenId ->
                            tokenizer.decode(longArrayOf(tokenId.toLong())).trim()
                        }
                    }

                    override fun getBatchifier(): Batchifier? = null // We handle batching manually or process single input
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
                        val result = predictor.predict(sentence).trim()
                        val topPrediction = result.substringBefore(", ")
                        val completed = sentence.replace("[MASK]", topPrediction)
                        println("Input: $sentence")
                        println("Prediction (top-$topK): [$result]")
                        println("Completed: $completed")
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
