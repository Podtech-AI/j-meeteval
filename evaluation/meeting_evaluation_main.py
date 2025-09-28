# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 会議関連指示の評価メイン（IFEval構造準拠）"""

import os
import sys
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meeting_evaluation_lib

_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "jMeet-Eval入力データへのパス", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "jMeet-Eval入力応答データへのパス", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "jMeet-Eval推論と評価結果の出力ディレクトリ",
    required=True,
)

_VERBOSE = flags.DEFINE_bool(
    "verbose", False, "詳細なログ出力を有効にする"
)

_STRICT = flags.DEFINE_bool(
    "strict", True, "厳密評価モードを使用"
)

def read_prompt_to_response_dict(input_response_data_file):
    """応答データファイルを読み込み"""
    if not input_response_data_file or not os.path.exists(input_response_data_file):
        return {}
    
    import json
    responses = {}
    with open(input_response_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses[data["prompt"]] = data["response"]
    return responses

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("引数が多すぎます。")

    # 出力ディレクトリを作成
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    
    # 入力データを読み込み
    logging.info("jMeet-Eval入力データを読み込み中...")
    inputs = meeting_evaluation_lib.read_meeting_prompt_list(_INPUT_DATA.value)
    logging.info(f"読み込み完了: {len(inputs)}件の入力データ")
    
    # 応答データを読み込み
    prompt_to_response = {}
    if _INPUT_RESPONSE_DATA.value:
        logging.info("応答データを読み込み中...")
        prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)
        logging.info(f"読み込み完了: {len(prompt_to_response)}件の応答データ")
    
    # 評価器を初期化
    evaluator = meeting_evaluation_lib.MeetingEvaluator(
        strict=_STRICT.value, 
        verbose=_VERBOSE.value
    )
    
    # 評価実行
    logging.info("jMeet-Eval評価を実行中...")
    outputs = []
    
    for input_example in inputs:
        # 応答を取得（応答データがない場合はモック応答を使用）
        if input_example.prompt in prompt_to_response:
            response = prompt_to_response[input_example.prompt]
        else:
            response = "これはモック応答です。実際の評価には応答データが必要です。"
            
        # 評価実行
        output = evaluator.evaluate_response(input_example, response)
        outputs.append(output)
    
    # メトリクス計算
    logging.info("メトリクスを計算中...")
    metrics = meeting_evaluation_lib.calculate_meeting_metrics(outputs)
    
    # 結果をファイルに出力
    results_file = os.path.join(_OUTPUT_DIR.value, "jmeet_eval_results_strict.jsonl")
    meeting_evaluation_lib.write_meeting_results(outputs, results_file)
    
    # メトリクスを出力
    metrics_file = os.path.join(_OUTPUT_DIR.value, "jmeet_metrics.json")
    import json
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 詳細レポートを生成
    report_file = os.path.join(_OUTPUT_DIR.value, "jmeet_evaluation_report.json")
    meeting_evaluation_lib.generate_evaluation_report(metrics, report_file)
    
    # 結果を表示
    logging.info("=== jMeet-Eval 評価結果 ===")
    logging.info(f"プロンプトレベル精度: {metrics['prompt_level_strict_accuracy']:.3f}")
    logging.info(f"指示レベル精度: {metrics['instruction_level_strict_accuracy']:.3f}")
    
    logging.info("=== カテゴリ別精度 ===")
    for key, value in metrics.items():
        if key.endswith("_accuracy") and not key.startswith(("prompt_", "instruction_")):
            logging.info(f"{key}: {value:.3f}")
    
    logging.info(f"詳細結果: {results_file}")
    logging.info(f"メトリクス: {metrics_file}")
    logging.info(f"評価レポート: {report_file}")

if __name__ == "__main__":
    app.run(main)