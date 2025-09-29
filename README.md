# J-MeetEval

J-MeetEvalは、日本語の会議関連タスクにおける指示従属性を評価するための包括的なベンチマークです。IFEvalフレームワークと同じ構造を採用し、実際の企業環境における会議シナリオに特化した10の指示カテゴリで構成されています。

エンドツーエンド評価パイプライン（`jmeet_eval_pipeline.py`）により、応答生成から評価まで一括実行でき、デフォルトでは2例の素早いテストが可能です。

## 特徴

- **🚀 パイプライン実行**: ワンコマンドで応答生成から評価まで完結
- **⚡ 高速テスト**: デフォルト2例で素早い動作確認
- **🎯 実務特化**: 実際の企業会議で使用される10種類の指示カテゴリ
- **📝 出力制約**: フィラー除去、句点制御、文末制約など実用的な出力制約
- **🌐 言語対応**: 日本語要約、英訳など多言語対応の指示
- **📊 分量制御**: 文字数制限、文数制限、割合指定による分量管理
- **📋 形式統一**: テンプレート充填、箇条書き、表形式など多様な出力形式
- **🔗 IFEval準拠**: IFEvalと同じディレクトリ構造とAPI設計

## ディレクトリ構造（IFEval準拠）

```
jmeet-eval/
├── README.md                           # このファイル
├── jmeet_eval_pipeline.py              # エンドツーエンド評価パイプライン（推奨）
├── meeting_instructions.py             # 10種類の会議指示カテゴリクラス
├── meeting_instructions_registry.py    # 指示レジストリシステム
├── data/                               # 評価データ
│   ├── jmeet_input_data.jsonl          # 入力データ（10例）
│   └── output/                         # 評価結果出力
├── evaluation/                         # 評価システム
│   ├── meeting_evaluation_lib.py       # 評価ライブラリ
│   ├── meeting_evaluation_main.py      # 評価実行メイン
│   └── test_meeting_instructions.py    # テストスイート
└── generate/                           # データ生成システム
    └── generate_meeting_responses.py    # 応答生成
```

## クイックスタート

### 1. エンドツーエンド評価（推奨）
```bash
cd jmeet-eval

# デフォルト設定（2例をHuggingFaceモデルで評価）
python3 jmeet_eval_pipeline.py

# 全10例を評価
python3 jmeet_eval_pipeline.py --num_examples -1

# Claude APIで5例を詳細ログ付きで評価
python3 jmeet_eval_pipeline.py --model claude-3-5-sonnet --num_examples 5 --verbose
```

### 2. 個別実行（上級者向け）

#### 基本テスト実行
```bash
python3 evaluation/test_meeting_instructions.py
```

#### 応答生成のみ
```bash
# HuggingFaceモデルで応答生成
python3 generate/generate_meeting_responses.py \
  --input_data=data/jmeet_input_data.jsonl \
  --output=data/jmeet_response_data_hf.jsonl \
  --model=sbintuitions/sarashina2.2-1b-instruct-v0.1 \
  --device=cpu

# Claude APIで応答生成
python3 generate/generate_meeting_responses.py \
  --input_data=data/jmeet_input_data.jsonl \
  --output=data/jmeet_response_data_claude.jsonl \
  --model=claude-3-5-sonnet
```

#### 評価のみ
```bash
# 基本評価
python3 evaluation/meeting_evaluation_main.py \
  --input_data=data/jmeet_input_data.jsonl \
  --input_response_data=data/jmeet_response_data_hf.jsonl \
  --output_dir=data/output

# 詳細評価
python3 evaluation/meeting_evaluation_main.py \
  --input_data=data/jmeet_input_data.jsonl \
  --input_response_data=data/jmeet_response_data_claude.jsonl \
  --output_dir=data/output \
  --verbose \
  --strict
```

## 評価指示カテゴリ

jMeet-Evalは10のカテゴリで実用的な会議関連指示を評価します：

### 1. 出力制約 (Output Constraints) - 4指示
- **フィラー除去**: あのー、えっと等のフィラー語の除去
- **末尾制約**: 「以上」等の不要な文末表現の除去
- **句点制御**: 句点（。）の除去制御
- **議事録限定**: 議事録以外の余計な出力の制限

### 2. 言語指定・翻訳 (Language and Translation) - 2指示
- **日本語要約**: 要約を日本語で記載
- **英語翻訳**: 指定部分の英訳提供

### 3. 分量指定 (Length Control) - 3指示
- **文字数制限**: 指定文字数以内または範囲での記載
- **割合指定**: 元文書の指定割合での要約
- **文数制限**: 指定文数以内での記載、文字数制限との組み合わせ

### 4. 付記 (Annotations) - 2指示
- **末尾付記**: 指定文言の末尾追加
- **冒頭付記**: 指定文言の冒頭明記

### 5. テンプレート充填 (Template Filling) - 1指示
- **テンプレート形式**: 指定テンプレートへの情報記入

### 6. 箇条書き・フォーマット (Formatting) - 4指示
- **箇条書き**: 指定点数での箇条書き形式
- **章立て構成**: 指定セクション順での構成
- **Q&A形式**: 質問と回答のペア形式
- **必須項目**: 指定項目を含む記載

### 7. 内容評価 (Content Evaluation) - 2指示
- **評価記載**: A/B/C評価の記載
- **5段階評価**: ★マークまたは数字での5段階評価

### 8. JSON出力 (JSON Output) - 1指示
- **JSON形式**: 指定スキーマでのJSON出力

### 9. 文体・敬語 (Writing Style) - 2指示
- **文体統一**: です・ます調、である調等の統一
- **数値形式**: 全角・半角数字の統一

### 10. 表形式 (Table Format) - 1指示
- **Markdown表**: 指定カラムでのMarkdown表出力

## 主要機能

### 1. エンドツーエンド評価パイプライン
jmeet_eval_pipeline.pyによる一括評価：

- **ワンコマンド実行**: 応答生成から評価まで自動実行
- **デフォルト2例**: 素早いテストのためデフォルトは2例
- **タイムスタンプ付き出力**: 実行ごとに整理された結果
- **詳細サマリー**: 人が読みやすい評価レポート

### 2. 応答生成システム
generate_meeting_responses.pyは、多様なモデルに対応：

- **HuggingFaceモデル**: ローカルモデルでの応答生成
- **Claude API**: 高品質な応答生成
- **バッチ処理**: 大量データの効率的処理
- **エラーハンドリング**: 堅牢な処理継続

### 3. 評価システム
meeting_evaluation_lib.pyによる詳細な評価：

- **指示レベル評価**: 各指示の個別成功率測定
- **プロンプトレベル評価**: 全指示の総合成功率測定
- **カテゴリ別分析**: 10カテゴリ別の詳細分析
- **エラー分析**: 失敗パターンの詳細レポート

### 4. 複雑度レベル対応
10例のデータには多様な複雑度が含まれます：
- **単一指示**: 3例（基本レベル）
- **2つの指示**: 3例（中級レベル）
- **3つの指示**: 4例（上級レベル）

## データ形式

### 入力データ (JSONL形式)
```json
{
  "key": 1,
  "instruction_id_list": ["output_constraint:filler_removal"],
  "prompt": "あのー、今日の会議では、えっと、新しいマーケティング施策について話し合いました。",
  "kwargs": [{}]
}
```

### 応答データ (JSONL形式)
```json
{
  "prompt": "あのー、今日の会議では、えっと、新しいマーケティング施策について話し合いました。",
  "response": "今日の会議では、新しいマーケティング施策について話し合いました。"
}
```

## 評価メトリクス

### 基本メトリクス
```json
{
  "prompt_level_strict_accuracy": 0.72,
  "instruction_level_strict_accuracy": 0.83,
  "output_constraint_accuracy": 0.85,
  "language_translation_accuracy": 0.78,
  "length_control_accuracy": 0.69
}
```

### 詳細分析
- **カテゴリ別分析**: 10カテゴリの個別成功率比較
- **複雑度別分析**: 単一・複合指示の成功率比較
- **エラーパターン分析**: 指示別の失敗パターン分類
- **パフォーマンス統計**: 実行時間・効率性分析

## API設計（IFEval互換）

### パイプライン使用（推奨）
```python
from jmeet_eval_pipeline import JMeetEvalPipeline

# パイプラインを初期化
pipeline = JMeetEvalPipeline(
    model="claude-3-5-sonnet",
    device="cpu",
    verbose=True,
    strict=True
)

# 評価実行（デフォルト2例）
eval_results = pipeline.run(num_examples=2)

# 全データ評価
eval_results = pipeline.run(num_examples=-1)
```

### 個別モジュール使用
```python
from generate.generate_meeting_responses import MeetingResponseGenerator
from evaluation import meeting_evaluation_lib

# 応答ジェネレーターを初期化
generator = MeetingResponseGenerator("claude-3-5-sonnet")

# プロンプトに対する応答生成
response = generator.generate_response("あのー、今日の会議では...")

# 評価器を初期化
evaluator = meeting_evaluation_lib.MeetingEvaluator(strict=True, verbose=True)

# 入力データを読み込み
inputs = meeting_evaluation_lib.read_meeting_prompt_list("data/jmeet_input_data.jsonl")

# 評価実行
for input_example in inputs:
    result = evaluator.evaluate_response(input_example, response)
```

### コマンドライン使用
```bash
# パイプライン実行（推奨）
python3 jmeet_eval_pipeline.py --model claude-3-5-sonnet --num_examples 5

# 個別実行
python3 generate/generate_meeting_responses.py \
  --input_data=data/jmeet_input_data.jsonl \
  --output=data/jmeet_response_data_claude.jsonl \
  --model=claude-3-5-sonnet

python3 evaluation/meeting_evaluation_main.py \
  --input_data=data/jmeet_input_data.jsonl \
  --input_response_data=data/jmeet_response_data_claude.jsonl \
  --output_dir=data/output
```

## 拡張性

新しい指示タイプの追加方法：

1. `meeting_instructions.py`に新しい指示クラスを実装
2. `meeting_instructions_registry.py`にレジストリ登録
3. `evaluation/test_meeting_instructions.py`にテスト追加
4. `data/jmeet_input_data.jsonl`に対応するデータを追加


## 依存関係

- Python 3.8+
- absl-py（評価フラグシステム用）
- transformers（HuggingFaceモデル用、オプション）
- anthropic（Claude API用、オプション）

## ライセンス

検討中

## 貢献・サポート

バグ報告、機能提案、改善点についてはGitHubのIssueページでお知らせください。

---

**J-MeetEval**: IFEvalと同じ設計思想で構築された、日本語会議タスクの実用的な指示従属性評価ベンチマーク