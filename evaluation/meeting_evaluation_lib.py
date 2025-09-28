# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 会議関連指示の評価ライブラリ（拡張版）"""

import collections
import dataclasses
import json
import sys
import os
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import re

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import meeting_instructions_registry
from meeting_instructions import MeetingInstruction

@dataclasses.dataclass
class MeetingInputExample:
    """会議評価の入力例"""
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]
    metadata: Optional[Dict[str, Any]] = None

@dataclasses.dataclass  
class MeetingOutputExample:
    """会議評価の出力例"""
    key: int
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[List[str]] = None

class MeetingEvaluator:
    """会議指示の評価を行うメインクラス"""
    
    def __init__(self, strict: bool = True, verbose: bool = False):
        self.strict = strict
        self.verbose = verbose
        self.instruction_registry = meeting_instructions_registry.MEETING_INSTRUCTION_DICT
        self._cache = {}
        
    def evaluate_single_instruction(self, 
                                  instruction_id: str, 
                                  prompt: str, 
                                  response: str, 
                                  kwargs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """単一の会議指示を評価"""
        try:
            # キャッシュチェック
            cache_key = f"{instruction_id}:{hash(response)}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # 会議指示クラスを取得
            instruction_class = self.instruction_registry.get(instruction_id)
            if not instruction_class:
                return False, f"Unknown instruction ID: {instruction_id}"
                
            # インスタンスを作成
            instruction_instance = instruction_class(instruction_id)
            
            # kwargsをインスタンスに設定（チェック時に使用）
            instruction_instance._cached_kwargs = kwargs
            
            # 評価実行
            result = instruction_instance.check_following(response)
            
            # 詳細なエラー情報を取得（verbose時）
            error_detail = None
            if not result and self.verbose:
                error_detail = self._generate_error_detail(
                    instruction_instance, response, kwargs
                )
            
            # キャッシュに保存
            self._cache[cache_key] = (result, error_detail)
            
            return result, error_detail
            
        except Exception as e:
            error_msg = f"Evaluation error - instruction_id: {instruction_id}, error: {str(e)}"
            if self.verbose:
                print(error_msg)
            return False, error_msg
    
    def _generate_error_detail(self, 
                             instruction: MeetingInstruction, 
                             response: str, 
                             kwargs: Dict[str, Any]) -> str:
        """エラーの詳細情報を生成"""
        expected = instruction.generate_instruction(**kwargs)
        return f"Expected format: {expected[:100]}... | Response length: {len(response)} chars"
    
    def evaluate_response(self, 
                         input_example: MeetingInputExample, 
                         response: str) -> MeetingOutputExample:
        """会議応答を評価"""
        start_time = time.time()
        follow_instruction_list = []
        error_details = []
        
        # 各指示を評価
        for i, instruction_id in enumerate(input_example.instruction_id_list):
            kwargs = input_example.kwargs[i] if i < len(input_example.kwargs) else {}
            
            is_following, error_detail = self.evaluate_single_instruction(
                instruction_id, 
                input_example.prompt, 
                response, 
                kwargs
            )
            
            follow_instruction_list.append(is_following)
            if error_detail:
                error_details.append(f"{instruction_id}: {error_detail}")
        
        follow_all_instructions = all(follow_instruction_list)
        execution_time = time.time() - start_time
        
        return MeetingOutputExample(
            key=input_example.key,
            instruction_id_list=input_example.instruction_id_list,
            prompt=input_example.prompt,
            response=response,
            follow_all_instructions=follow_all_instructions,
            follow_instruction_list=follow_instruction_list,
            execution_time=execution_time,
            metadata=input_example.metadata,
            error_details=error_details if error_details else None
        )

def read_meeting_prompt_list(input_jsonl_filename: str) -> List[MeetingInputExample]:
    """会議評価用JSONLファイルから入力を読み込み"""
    inputs = []
    with open(input_jsonl_filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    input_data = json.loads(line)
                    inputs.append(MeetingInputExample(
                        key=input_data["key"],
                        instruction_id_list=input_data["instruction_id_list"],
                        prompt=input_data["prompt"],
                        kwargs=input_data.get("kwargs", []),
                        metadata=input_data.get("metadata", {})
                    ))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                except KeyError as e:
                    print(f"Missing required field in line {line_num}: {e}")
    return inputs

def calculate_meeting_metrics(output_examples: List[MeetingOutputExample]) -> Dict[str, Any]:
    """会議評価メトリクスを計算（拡張版）"""
    if not output_examples:
        return {"error": "No output examples provided"}
    
    total_prompts = len(output_examples)
    
    # 基本メトリクス
    prompt_level_accuracy = sum(ex.follow_all_instructions for ex in output_examples) / total_prompts
    
    # 指示レベルの詳細統計
    instruction_results = []
    instruction_stats = collections.defaultdict(lambda: {"total": 0, "passed": 0})
    category_results = collections.defaultdict(list)
    
    # 実行時間統計
    execution_times = [ex.execution_time for ex in output_examples]
    
    for example in output_examples:
        for i, instruction_id in enumerate(example.instruction_id_list):
            is_following = example.follow_instruction_list[i]
            instruction_results.append(is_following)
            
            # 指示別統計
            instruction_stats[instruction_id]["total"] += 1
            if is_following:
                instruction_stats[instruction_id]["passed"] += 1
            
            # カテゴリ別の結果を記録
            for cat_name, cat_prefix in meeting_instructions_registry.MEETING_CATEGORIES.items():
                if instruction_id.startswith(cat_prefix):
                    category_results[cat_name].append(is_following)
                    break
    
    instruction_level_accuracy = sum(instruction_results) / len(instruction_results) if instruction_results else 0
    
    # カテゴリ別精度計算
    category_accuracies = {}
    for category, results in category_results.items():
        if results:
            category_accuracies[f"{category}_accuracy"] = sum(results) / len(results)
        else:
            category_accuracies[f"{category}_accuracy"] = 0
    
    # 指示別成功率
    instruction_success_rates = {}
    for inst_id, stats in instruction_stats.items():
        if stats["total"] > 0:
            instruction_success_rates[inst_id] = stats["passed"] / stats["total"]
    
    # 複雑度別の分析
    complexity_analysis = analyze_by_complexity(output_examples)
    
    # エラー分析
    error_analysis = analyze_errors(output_examples)
    
    return {
        # 基本メトリクス
        "prompt_level_strict_accuracy": prompt_level_accuracy,
        "instruction_level_strict_accuracy": instruction_level_accuracy,
        
        # カテゴリ別メトリクス
        **category_accuracies,
        
        # 詳細統計
        "total_prompts": total_prompts,
        "total_instructions": len(instruction_results),
        "instruction_success_rates": instruction_success_rates,
        
        # 複雑度分析
        "complexity_analysis": complexity_analysis,
        
        # エラー分析
        "error_analysis": error_analysis,
        
        # パフォーマンス統計
        "performance_stats": {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times)
        },
        
        # タイムスタンプ
        "evaluation_timestamp": datetime.now().isoformat()
    }

def analyze_by_complexity(output_examples: List[MeetingOutputExample]) -> Dict[str, Any]:
    """複雑度別の分析"""
    complexity_groups = {
        "single_instruction": [],
        "double_instruction": [],
        "multiple_instruction": []
    }
    
    for example in output_examples:
        num_instructions = len(example.instruction_id_list)
        if num_instructions == 1:
            complexity_groups["single_instruction"].append(example.follow_all_instructions)
        elif num_instructions == 2:
            complexity_groups["double_instruction"].append(example.follow_all_instructions)
        else:
            complexity_groups["multiple_instruction"].append(example.follow_all_instructions)
    
    analysis = {}
    for group_name, results in complexity_groups.items():
        if results:
            analysis[group_name] = {
                "count": len(results),
                "accuracy": sum(results) / len(results),
                "percentage": len(results) / len(output_examples) * 100
            }
        else:
            analysis[group_name] = {
                "count": 0,
                "accuracy": 0,
                "percentage": 0
            }
    
    return analysis

def analyze_errors(output_examples: List[MeetingOutputExample]) -> Dict[str, Any]:
    """エラーパターンの分析"""
    error_patterns = collections.defaultdict(int)
    error_by_instruction = collections.defaultdict(list)
    
    for example in output_examples:
        if example.error_details:
            for error in example.error_details:
                # エラーパターンを抽出
                if "Expected format" in error:
                    error_patterns["format_mismatch"] += 1
                elif "Unknown instruction" in error:
                    error_patterns["unknown_instruction"] += 1
                elif "error" in error.lower():
                    error_patterns["execution_error"] += 1
                else:
                    error_patterns["other"] += 1
                
                # 指示別エラー記録
                inst_id = error.split(":")[0]
                error_by_instruction[inst_id].append(error)
    
    return {
        "error_patterns": dict(error_patterns),
        "total_errors": sum(error_patterns.values()),
        "most_common_error": max(error_patterns.items(), key=lambda x: x[1])[0] if error_patterns else None,
        "instructions_with_most_errors": sorted(
            [(inst, len(errors)) for inst, errors in error_by_instruction.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }

def write_meeting_results(output_examples: List[MeetingOutputExample], 
                         output_filename: str,
                         include_metadata: bool = True):
    """会議評価結果をJSONLファイルに出力（拡張版）"""
    with open(output_filename, "w", encoding="utf-8") as f:
        for example in output_examples:
            result = {
                "key": example.key,
                "instruction_id_list": example.instruction_id_list,
                "prompt": example.prompt,
                "response": example.response,
                "follow_all_instructions": example.follow_all_instructions,
                "follow_instruction_list": example.follow_instruction_list,
                "execution_time": example.execution_time
            }
            
            if include_metadata:
                if example.metadata:
                    result["metadata"] = example.metadata
                if example.error_details:
                    result["error_details"] = example.error_details
            
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def generate_evaluation_report(metrics: Dict[str, Any], output_file: str):
    """詳細な評価レポートを生成"""
    report = {
        "summary": {
            "prompt_accuracy": metrics["prompt_level_strict_accuracy"],
            "instruction_accuracy": metrics["instruction_level_strict_accuracy"],
            "total_evaluated": metrics["total_prompts"]
        },
        "category_performance": {
            cat: metrics.get(f"{cat}_accuracy", 0)
            for cat in meeting_instructions_registry.MEETING_CATEGORIES.keys()
        },
        "complexity_breakdown": metrics.get("complexity_analysis", {}),
        "error_summary": metrics.get("error_analysis", {}),
        "performance": metrics.get("performance_stats", {}),
        "detailed_instruction_rates": metrics.get("instruction_success_rates", {}),
        "timestamp": metrics.get("evaluation_timestamp")
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)