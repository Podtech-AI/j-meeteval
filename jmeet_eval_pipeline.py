#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: エンドツーエンド評価パイプライン

使用例:
    # HuggingFaceモデルで2例を評価（デフォルト）
    python jmeet_eval_pipeline.py
    
    # Claude APIで全データを評価
    python jmeet_eval_pipeline.py --model claude-3-5-sonnet --num_examples -1
    
    # 特定の数の例を評価
    python jmeet_eval_pipeline.py --num_examples 5 --verbose
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate.generate_meeting_responses import MeetingResponseGenerator
from evaluation.meeting_evaluation_lib import (
    MeetingEvaluator, 
    read_meeting_prompt_list,
    calculate_meeting_metrics,
    MeetingInputExample
)

class JMeetEvalPipeline:
    """jMeet-Eval評価パイプライン"""
    
    def __init__(self, 
                 model: str = "sbintuitions/sarashina2.2-1b-instruct-v0.1",
                 device: str = "cpu",
                 verbose: bool = False,
                 strict: bool = True):
        """
        Args:
            model: 使用するモデル名
            device: 計算デバイス（cpu/cuda）
            verbose: 詳細ログ出力
            strict: 厳密評価モード
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        self.strict = strict
        
        # ファイルパス
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_data_path = os.path.join(self.base_dir, "data", "jmeet_input_data.jsonl")
        self.output_dir = os.path.join(self.base_dir, "data", "output")
        
        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # タイムスタンプ付き出力ファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.response_file = os.path.join(self.output_dir, f"responses_{timestamp}.jsonl")
        self.eval_results_file = os.path.join(self.output_dir, f"eval_results_{timestamp}.json")
        self.summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.txt")
        
    def load_input_data(self, num_examples: int = 2) -> List[Dict]:
        """入力データの読み込み
        
        Args:
            num_examples: 読み込む例の数（-1で全て）
            
        Returns:
            入力データのリスト
        """
        print(f"\n=== 入力データ読み込み ===")
        
        # MeetingInputExampleオブジェクトとして読み込み
        all_examples = read_meeting_prompt_list(self.input_data_path)
        
        # 指定数のみ取得
        if num_examples == -1 or num_examples >= len(all_examples):
            examples = all_examples
        else:
            examples = all_examples[:num_examples]
        
        print(f"読み込んだ例: {len(examples)}/{len(all_examples)}")
        
        if self.verbose:
            for ex in examples:
                print(f"\nKey {ex.key}: {len(ex.instruction_id_list)} instructions")
                print(f"Instructions: {', '.join(ex.instruction_id_list)}")
        
        return examples
    
    def generate_responses(self, examples: List[MeetingInputExample]) -> List[Dict]:
        """応答生成
        
        Args:
            examples: 入力例のリスト
            
        Returns:
            応答データのリスト
        """
        print(f"\n=== 応答生成開始 ===")
        print(f"モデル: {self.model}")
        print(f"デバイス: {self.device}")
        
        # 応答生成器を初期化
        generator = MeetingResponseGenerator(self.model, self.device)
        
        responses = []
        start_time = time.time()
        
        for i, example in enumerate(examples):
            print(f"\n[{i+1}/{len(examples)}] Key {example.key} の応答生成中...")
            
            try:
                # 応答生成
                response = generator.generate_response(example.prompt)
                
                # 応答データを作成
                response_data = {
                    "key": example.key,
                    "prompt": example.prompt,
                    "response": response,
                    "instruction_id_list": example.instruction_id_list
                }
                
                responses.append(response_data)
                
                if self.verbose:
                    print(f"応答文字数: {len(response)}")
                    print(f"応答プレビュー: {response[:100]}...")
                    
            except Exception as e:
                print(f"エラー発生: {e}")
                # エラー時は空応答を記録
                responses.append({
                    "key": example.key,
                    "prompt": example.prompt,
                    "response": "",
                    "instruction_id_list": example.instruction_id_list,
                    "error": str(e)
                })
        
        # 応答をファイルに保存
        with open(self.response_file, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        
        generation_time = time.time() - start_time
        print(f"\n応答生成完了: {generation_time:.2f}秒")
        print(f"応答を保存: {self.response_file}")
        
        return responses
    
    def evaluate_responses(self, 
                          examples: List[MeetingInputExample], 
                          responses: List[Dict]) -> Dict:
        """応答評価
        
        Args:
            examples: 入力例のリスト
            responses: 応答データのリスト
            
        Returns:
            評価結果の辞書
        """
        print(f"\n=== 評価開始 ===")
        print(f"評価モード: {'厳密' if self.strict else '寛容'}")
        
        # 評価器を初期化
        evaluator = MeetingEvaluator(strict=self.strict, verbose=self.verbose)
        
        # 評価結果を格納
        outputs = []
        start_time = time.time()
        
        # 応答をキーでインデックス化
        response_map = {r["key"]: r["response"] for r in responses}
        
        for i, example in enumerate(examples):
            print(f"\n[{i+1}/{len(examples)}] Key {example.key} を評価中...")
            
            # 対応する応答を取得
            response = response_map.get(example.key, "")
            
            # 評価実行
            result = evaluator.evaluate_response(example, response)
            outputs.append(result)
            
            if self.verbose:
                print(f"全指示に従った: {result.follow_all_instructions}")
                print(f"個別結果: {result.follow_instruction_list}")
                if result.error_details:
                    print(f"エラー詳細: {result.error_details}")
        
        # メトリクス計算
        metrics = calculate_meeting_metrics(outputs)
        
        evaluation_time = time.time() - start_time
        print(f"\n評価完了: {evaluation_time:.2f}秒")
        
        # 結果を保存
        eval_results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "num_examples": len(examples),
            "metrics": metrics,
            "details": [
                {
                    "key": output.key,
                    "instruction_id_list": output.instruction_id_list,
                    "follow_all_instructions": output.follow_all_instructions,
                    "follow_instruction_list": output.follow_instruction_list,
                    "error_details": output.error_details
                }
                for output in outputs
            ]
        }
        
        with open(self.eval_results_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        print(f"評価結果を保存: {self.eval_results_file}")
        
        return eval_results
    
    def generate_summary(self, eval_results: Dict) -> None:
        """評価サマリーの生成
        
        Args:
            eval_results: 評価結果の辞書
        """
        print(f"\n=== サマリー生成 ===")
        
        metrics = eval_results["metrics"]
        
        summary_lines = [
            "=" * 60,
            "jMeet-Eval 評価サマリー",
            "=" * 60,
            f"実行日時: {eval_results['timestamp']}",
            f"使用モデル: {eval_results['model']}",
            f"評価例数: {eval_results['num_examples']}",
            "",
            "【全体メトリクス】",
            f"プロンプトレベル精度: {metrics['prompt_level_strict_accuracy']:.2%}",
            f"指示レベル精度: {metrics['instruction_level_strict_accuracy']:.2%}",
            "",
            "【カテゴリ別精度】"
        ]
        
        # カテゴリ別精度
        category_metrics = [(k, v) for k, v in metrics.items() 
                          if k.endswith('_accuracy') 
                          and not k.startswith(('prompt_level', 'instruction_level'))]
        
        for category, accuracy in sorted(category_metrics):
            category_name = category.replace('_accuracy', '')
            summary_lines.append(f"{category_name}: {accuracy:.2%}")
        
        # エラーパターン分析
        error_count = sum(1 for detail in eval_results['details'] 
                         if not detail['follow_all_instructions'])
        
        summary_lines.extend([
            "",
            "【エラー分析】",
            f"失敗例数: {error_count}/{eval_results['num_examples']}",
            ""
        ])
        
        # 失敗した指示の詳細
        if error_count > 0:
            summary_lines.append("失敗した例:")
            for detail in eval_results['details']:
                if not detail['follow_all_instructions']:
                    summary_lines.append(f"  Key {detail['key']}: {detail['instruction_id_list']}")
        
        summary_lines.extend([
            "",
            "=" * 60,
            f"応答ファイル: {self.response_file}",
            f"評価結果ファイル: {self.eval_results_file}",
            f"サマリーファイル: {self.summary_file}",
            "=" * 60
        ])
        
        # サマリーを保存
        summary_text = '\n'.join(summary_lines)
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # コンソールに出力
        print(summary_text)
    
    def run(self, num_examples: int = 2) -> Dict:
        """パイプライン実行
        
        Args:
            num_examples: 評価する例の数（-1で全て）
            
        Returns:
            評価結果の辞書
        """
        print(f"\n{'=' * 60}")
        print(f"jMeet-Eval Pipeline 開始")
        print(f"{'=' * 60}")
        
        total_start_time = time.time()
        
        try:
            # 1. データ読み込み
            examples = self.load_input_data(num_examples)
            
            # 2. 応答生成
            responses = self.generate_responses(examples)
            
            # 3. 評価実行
            eval_results = self.evaluate_responses(examples, responses)
            
            # 4. サマリー生成
            self.generate_summary(eval_results)
            
            total_time = time.time() - total_start_time
            print(f"\n総実行時間: {total_time:.2f}秒")
            
            return eval_results
            
        except Exception as e:
            print(f"\nパイプライン実行中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="jMeet-Eval エンドツーエンド評価パイプライン"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sbintuitions/sarashina2.2-1b-instruct-v0.1',
        help='使用するモデル名（HuggingFaceモデルまたはclaude-*）'
    )
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=2,
        help='評価する例の数（-1で全て）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='計算デバイス'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログを出力'
    )
    
    parser.add_argument(
        '--no_strict',
        action='store_true',
        help='寛容な評価モードを使用'
    )
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = JMeetEvalPipeline(
        model=args.model,
        device=args.device,
        verbose=args.verbose,
        strict=not args.no_strict
    )
    
    try:
        pipeline.run(num_examples=args.num_examples)
    except KeyboardInterrupt:
        print("\n\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()