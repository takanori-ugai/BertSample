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
import ai.djl.nn.transformer.BertMaskedLanguageModelLoss
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.min
import kotlinx.coroutines.*

/**
 * BertMaskedLanguageModelBlock と BertMaskedLanguageModelLoss を使用した、
 * GPUアクセラレーション対応の BERT MLM 学習システム
 */
fun main() = runBlocking {
    val filePath = "train.txt"
    val batchSize = 10
    val maxSequenceLength = 128
    val numEpochs = 10

    // GPUが利用可能かチェックし、デバイスを決定
    val device = if (Engine.getInstance().gpuCount > 0) Device.gpu() else Device.cpu()
    println("使用デバイス: $device")

    if (!Files.exists(Paths.get(filePath))) {
        System.err.println("$filePath が見つかりません。")
        return@runBlocking
    }

    // 1. Tokenizer, Manager, Model の初期化
    HuggingFaceTokenizer.newInstance("bert-base-uncased").use { tokenizer ->
        NDManager.newBaseManager(device).use { manager ->
            Model.newInstance("bert-mlm", device).use { model ->

                val vocabSize = 30522
                val embeddingSize = 768

                // 2. BertMaskedLanguageModelBlock の構築
                val bertBlock = BertBlock.builder()
                    .setTokenDictionarySize(vocabSize)
                    .optTransformerBlockCount(4)
                    .optAttentionHeadCount(8)
                    .optEmbeddingSize(embeddingSize)
                    .optMaxSequenceLength(maxSequenceLength)
                    .build()

                val mlmBlock = BertMaskedLanguageModelBlock(bertBlock, Activation::gelu)

                // カスタムブロックの定義
                val combinedBlock = object : SequentialBlock() {
                    init {
                        add(bertBlock)
                        add(mlmBlock)
                    }

                    override fun forwardInternal(
                        ps: ai.djl.training.ParameterStore,
                        inputs: NDList,
                        training: Boolean,
                        params: ai.djl.util.PairList<String, Any>?
                    ): NDList {
                        // inputs: [tokenIds, typeIds, masks, maskedIndices]
                        val tokenIds = inputs[0]
                        val typeIds = inputs[1]
                        val masks = inputs[2]
                        val maskedIndices = inputs[3]
                        
                        // 1. BertBlock (Encoder) の実行
                        val bertInput = NDList(tokenIds, typeIds, masks)
                        val bertOutput = bertBlock.forward(ps, bertInput, training)
                        val sequenceOutput = bertOutput[0] // [batch, seq_len, embedding_size]
                        
                        // 2. Embedding Table の取得
                        val embeddingParam = bertBlock.tokenEmbedding.parameters[0].value
                        val embeddingTable = ps.getValue(embeddingParam, tokenIds.device, training)
                        
                        // 3. BertMaskedLanguageModelBlock (MLM Head) の実行
                        val mlmInput = NDList(sequenceOutput, maskedIndices, embeddingTable)
                        val mlmOutput = mlmBlock.forward(ps, mlmInput, training)
                        
                        return mlmOutput // [log_probs]
                    }

                    override fun initializeChildBlocks(manager: NDManager, dataType: ai.djl.ndarray.types.DataType, vararg inputShapes: Shape) {
                        // inputShapes: [tokenIds, typeIds, masks, maskedIndices]
                        val tokenShape = inputShapes[0]
                        val typeShape = inputShapes[1]
                        val maskShape = inputShapes[2]
                        val maskedIndicesShape = inputShapes[3]
                        
                        // BertBlock の初期化
                        bertBlock.initialize(manager, dataType, tokenShape, typeShape, maskShape)
                        
                        // BertBlock の出力形状を取得
                        val bertOutputShapes = bertBlock.getOutputShapes(arrayOf(tokenShape, typeShape, maskShape))
                        val sequenceOutputShape = bertOutputShapes[0] // [batch, seq_len, embedding_size]
                        
                        // Embedding Table の形状 (vocab_size, embedding_size)
                        val embeddingTableShape = Shape(vocabSize.toLong(), embeddingSize.toLong())
                        
                        // BertMaskedLanguageModelBlock の初期化
                        // 期待される入力: [sequenceOutput, maskedIndices, embeddingTable]
                        mlmBlock.initialize(manager, dataType, sequenceOutputShape, maskedIndicesShape, embeddingTableShape)
                    }
                    
                    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
                        // 最終的な出力形状を返す (log_probs)
                        val batchSize = inputShapes[0].get(0)
                        val maskedCount = inputShapes[3].get(1)
                        return arrayOf(Shape(batchSize * maskedCount, vocabSize.toLong()))
                    }
                }

                model.block = combinedBlock

                // 3. 学習設定
                // BertMaskedLanguageModelLoss(int labelIdx, int maskIdx, int logProbsIdx)
                // labelIdx: ラベルのインデックス (0)
                // maskIdx: マスクのインデックス (-1 は無効なインデックスなのでエラーになる)
                // logProbsIdx: 予測確率のインデックス (0)
                // BertMaskedLanguageModelLoss は maskIdx を使って labels からマスクを作成しようとします。
                // labels.get(maskIdx) を呼び出すため、-1 だと IndexOutOfBoundsException になります。
                // しかし、今回の実装では labels 自体が既にマスクされたトークンIDのみを含んでおり、
                // マスク位置の情報 (maskedIndices) はモデル内で使用され、Loss計算時には不要です。
                // BertMaskedLanguageModelLoss は、入力として (labels, predictions) を受け取りますが、
                // 内部で labels.get(maskIdx) を呼ぶため、maskIdx は有効なインデックスでなければなりません。
                // つまり、BertMaskedLanguageModelLoss は、labels リストの中にマスク情報も含まれていることを期待しています。
                // しかし、現在の trainer.loss.evaluate(NDList(labelsND), outputs) では、
                // labels リストには labelsND (正解トークン) しか含まれていません。
                // したがって、maskIdx を指定することはできません。
                
                // 解決策:
                // BertMaskedLanguageModelLoss は、マスク位置に基づいてロスを計算するためのものですが、
                // 今回のモデル (BertMaskedLanguageModelBlock) は既にマスク位置の予測のみを出力しています (gatherFromIndices 済み)。
                // したがって、出力は [batch * masked_count, vocab_size] となっており、
                // ラベルも [batch * masked_count] になっています (prepareData0 で調整済み)。
                // つまり、単純な SoftmaxCrossEntropyLoss で計算可能です。
                // BertMaskedLanguageModelLoss を使う必要はなく、むしろ使うとインデックスの問題が発生します。
                // ここでは、標準の Loss.softmaxCrossEntropyLoss を使用します。
                // ただし、パディングされた部分 (-1) を無視する必要があります。
                
                val loss = ai.djl.training.loss.Loss.softmaxCrossEntropyLoss("loss", 1.0f, -1, true, true)
                
                val config = DefaultTrainingConfig(loss)
                    .addTrainingListeners()

                model.newTrainer(config).use { trainer ->
                    // 初期化用の形状
                    val batchShape = Shape(batchSize.toLong(), maxSequenceLength.toLong())
                    val maskedIndicesShape = Shape(batchSize.toLong(), 20) 
                    
                    trainer.initialize(batchShape, batchShape, batchShape, maskedIndicesShape)

                    // 4. 学習データの読み込み
                    val allLines = Files.readAllLines(Paths.get(filePath))
                    val validLines = allLines.filter { it.trim().isNotEmpty() }.toMutableList()

                    val maskTokenId = tokenizer.encode("[MASK]").ids[0]
                    val padTokenId = tokenizer.encode("[PAD]").ids[0]

                    println("学習開始: ${validLines.size} 行のデータ, $numEpochs エポック")

                    // 5. エポックループ
                    for (epoch in 1..numEpochs) {
                        println("--- エポック $epoch / $numEpochs ---")
                        validLines.shuffle()

                        var epochLoss = 0f
                        var batchCount = 0

                        // ミニバッチ処理
                        for (i in validLines.indices step batchSize) {
                            val currentBatchSize = min(batchSize, validLines.size - i)

                            // 前処理（トークナイズ & マスク化）をコルーチンで並列実行
                            val batchData = coroutineScope {
                                (0 until currentBatchSize).map { b ->
                                    async(Dispatchers.Default) {
                                        val text = validLines[i + b]
                                        prepareData0(text, tokenizer, maxSequenceLength, vocabSize, maskTokenId, padTokenId)
                                    }
                                }.awaitAll()
                            }

                            // GPU/CPU でのモデル計算
                            trainer.newGradientCollector().use { gc ->
                                val inputIds = Array(currentBatchSize) { batchData[it].first } // tokenIds
                                val maskedIndices = Array(currentBatchSize) { batchData[it].second } // maskedIndices
                                val labels = Array(currentBatchSize) { batchData[it].third } // maskedTokens (labels)

                                val inputIndicesND = manager.create(inputIds)
                                val maskedIndicesND = manager.create(maskedIndices)
                                val labelsND = manager.create(labels)

                                val typeIndicesND = manager.zeros(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))
                                val maskIndicesND = manager.ones(Shape(currentBatchSize.toLong(), maxSequenceLength.toLong()))

                                // 順伝播: [tokenIds, typeIds, masks, maskedIndices]
                                val outputs = trainer.forward(NDList(inputIndicesND, typeIndicesND, maskIndicesND, maskedIndicesND))

                                // 損失計算: [labels], [log_probs]
                                // outputs は [batch * masked_count, vocab_size]
                                // labelsND は [batch, masked_count] なので、フラット化が必要
                                val flattenedLabels = labelsND.reshape(Shape(-1))
                                
                                // -1 のラベルを無視するためにフィルタリング
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
                        println("エポック $epoch 完了. 平均 Loss: ${if (batchCount > 0) epochLoss / batchCount else 0.0}")
                    }

                    // 6. モデルの保存
                    val modelDir = Paths.get("build/model")
                    if (!Files.exists(modelDir)) Files.createDirectories(modelDir)
                    model.save(modelDir, "bert-mlm")
                    println("モデルを保存しました: ${modelDir.toAbsolutePath()}")
                }
            }
        }
    }
}

/**
 * テキストをトークナイズし、BERT標準の 15% マスク処理を適用する（CPU実行）
 * 戻り値: Triple(inputIds, maskedIndices, maskedLabels)
 * maskedIndices: マスクされた位置のインデックス (フラット化されたインデックスではなく、シーケンス内の位置)
 * maskedLabels: マスクされた位置の正解トークンID
 */
fun prepareData0(
    text: String,
    tokenizer: HuggingFaceTokenizer,
    maxLen: Int,
    vocabSize: Int,
    maskId: Long,
    padId: Long
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
            // 15% の確率で予測対象（ラベル）にする
            maskedIndicesList.add(s.toLong())
            maskedLabelsList.add(originalId)

            val subR = random.nextDouble()
            inputIds[s] = when {
                subR < 0.8 -> maskId                          // 80% [MASK]
                subR < 0.9 -> random.nextInt(vocabSize).toLong() // 10% ランダム
                else -> originalId                             // 10% そのまま
            }
        } else {
            inputIds[s] = originalId
        }
    }

    // 最大長までパディング
    if (limit < maxLen) {
        for (s in limit until maxLen) {
            inputIds[s] = padId
        }
    }
    
    // maskedIndices と maskedLabels を固定長にパディングするか、あるいは可変長のまま扱うか。
    // DJL の gatherFromIndices は [batch, indices_per_seq] を期待するため、
    // バッチ内で長さを揃える必要があります。
    // ここでは簡易的に最大20個までとし、不足分はパディング (例えば 0) します。
    // ただし、0番目のトークン (CLS) がマスクされていない場合、0でパディングすると CLS を参照してしまいます。
    // 影響を最小限にするため、パディング部分は無視されるべきですが、
    // BertMaskedLanguageModelBlock は全てのインデックスに対して計算を行います。
    // 厳密にはマスク数を揃えるか、動的なグラフ構築が必要ですが、
    // ここでは簡易的に 0 で埋め、ラベルも -1 (無視) に設定します。
    
    val maxMasks = 20
    val maskedIndices = LongArray(maxMasks) { 0 }
    val maskedLabels = LongArray(maxMasks) { -1 } // Loss計算で無視されるはず
    
    for (i in 0 until min(maxMasks, maskedIndicesList.size)) {
        maskedIndices[i] = maskedIndicesList[i]
        maskedLabels[i] = maskedLabelsList[i]
    }

    return Triple(inputIds, maskedIndices, maskedLabels)
}