# BertSample

Kotlin と [Deep Java Library (DJL)](https://djl.ai/) を使った、BERT の Masked Language Modeling (MLM) のサンプルです。
`train.txt` のテキストを使って簡易的な BERT モデルを学習し、`[MASK]` の位置に入る単語を推論できます。

このプロジェクトは学習の仕組みを確認するためのサンプルです。事前学習済み BERT の重みは使用せず、モデルはランダムな重みから学習を開始します。

## Features

- Kotlin/JVM から DJL の BERT ブロックを利用
- CPU または GPU を自動選択
- BERT の Masked Language Modeling による学習
- 学習データのトークン化とマスク処理の並列化
- 学習済みモデルを使った `[MASK]` トークンの推論
- GPU が正しく認識されているかを確認する簡易プログラム

## Requirements

- JDK 17
- Gradle Wrapper（リポジトリに含まれています）
- GPU を使う場合は、対応する NVIDIA ドライバーと CUDA 環境
- 初回実行時に Hugging Face から `bert-base-uncased` の tokenizer ファイルを取得できるネットワーク接続

`build.gradle.kts` では DJL の CPU native runtime と CUDA 12.4 用 native runtime の両方を定義しています。GPU を使わない環境では、CUDA のバージョンや利用可能な DJL artifact に合わせて依存関係を調整してください。

## Build

```bash
./gradlew build
```

Windows の場合は次のコマンドを使用できます。

```bat
gradlew.bat build
```

## Run

`application` plugin の既定エントリポイントには `org.example.MainKt`（GPU確認用）を設定しています。次のコマンドで実行できます。

```bash
./gradlew run
```

`execute` タスクを使うと、`-PmainClass` で任意のエントリポイントを指定できます。

```bash
# BERT MLM の学習
./gradlew execute -PmainClass=org.example.BertMLMKt

# 学習済みモデルの推論
./gradlew execute -PmainClass=org.example.InferenceKt

# 並列学習版
./gradlew execute -PmainClass=org.example.BertMlmTrainingParallelKt

# 並列学習版に対応する推論
./gradlew execute -PmainClass=org.example.InferenceParallelKt
```

`-PmainClass` を省略した場合は `org.example.MainKt` が実行されます。IntelliJ IDEA などから各 `main` 関数を直接実行することもできます。

エントリポイントは次のとおりです。

| Kotlin ファイル | 実行クラス | 説明 |
| --- | --- | --- |
| `src/main/kotlin/Main.kt` | `org.example.MainKt` | DJL が GPU を認識しているか確認 |
| `src/main/kotlin/org/example/BertMLM.kt` | `org.example.BertMLMKt` | BERT MLM の学習（データ準備を並列化） |
| `src/main/kotlin/org/example/BertMlmTrainingParallel.kt` | `org.example.BertMlmTrainingParallelKt` | MLM block を使った学習（データ準備を並列化） |
| `src/main/kotlin/org/example/Inference.kt` | `org.example.InferenceKt` | `BertMLM.kt` のモデルを使った推論 |
| `src/main/kotlin/org/example/InferenceParallel.kt` | `org.example.InferenceParallelKt` | `BertMlmTrainingParallel.kt` のモデルを使った推論 |

### Recommended order

1. `org.example.MainKt` を実行して GPU の認識状態を確認します。
2. `org.example.BertMLMKt` または `org.example.BertMlmTrainingParallelKt` を実行して学習します。
3. 対応する推論クラスを実行します。

学習プログラムは `train.txt` を読み込み、学習済みモデルを次の場所に保存します。

```text
build/model/
```

推論プログラムはこのディレクトリの `bert-mlm` モデルを読み込みます。推論だけを先に実行した場合はモデルが存在しないため、エラーになります。

## Model configuration

現在のサンプルでは、主に次の設定を使用しています。

- Vocabulary size: `30,522`
- Embedding size: `768`
- Transformer blocks: `4`
- Attention heads: `8`
- Maximum sequence length: `128`
- Masking probability: `15%`
- Training batch size: `10`
- Number of epochs: `10`

これらの値は `BertMLM.kt` と `BertMlmTrainingParallel.kt` に定義されています。

## Training data

`train.txt` は 1 行 1 文のテキストファイルです。独自のデータを使う場合は、同じ形式で `train.txt` を置き換えてください。サンプルのデータ量は小さいため、学習結果はデモ用途に限られます。

## Notes

- 学習前のモデルはランダム初期化されるため、学習前に意味のある予測はできません。
- GPU が利用可能な場合は GPU を選択し、利用できない場合は CPU にフォールバックします。
- `Inference.kt` と `InferenceParallel.kt` はモデル構造が異なるため、対応する学習プログラムと組み合わせて使用してください。
- 推論時は各 `[MASK]` 位置について、logits の上位5候補（top-5）を表示します。
- CUDA 用 native runtime の依存関係は、実行環境の CUDA / NVIDIA ドライバーと互換性のあるものを選択してください。

## License

このリポジトリでは、現時点で個別のライセンスファイルは提供されていません。
