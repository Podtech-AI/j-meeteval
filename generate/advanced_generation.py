#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 既存データベースの高度な拡張生成システム"""

import json
import random
import argparse
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import itertools
from collections import defaultdict
import copy

# 親ディレクトリをパスに追加（jmeet-eval構造用）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import meeting_instructions_registry
from meeting_instructions import MeetingInstruction

class AdvancedMeetingDataGenerator:
    """既存データを基にした高度な会議評価データ生成器"""
    
    def __init__(self, base_data_file: str, seed: int = 42):
        self.base_data_file = base_data_file
        self.seed = seed
        random.seed(seed)
        self.registry = meeting_instructions_registry.MEETING_INSTRUCTION_DICT
        self.categories = meeting_instructions_registry.MEETING_CATEGORIES
        
        # 既存データを読み込み
        self.base_examples = self._load_base_data()
        
        # 拡張用の会議コンテキスト
        self.meeting_contexts = self._load_meeting_contexts()
        
        # 業界別コンテキスト
        self.industry_contexts = self._load_industry_contexts()
        
        # 難易度レベル
        self.difficulty_levels = ["basic", "intermediate", "advanced", "expert"]
        
        # プロンプト拡張パターン
        self.prompt_variations = self._load_prompt_variations()
        
    def _load_base_data(self) -> List[Dict[str, Any]]:
        """ベースとなるデータを読み込み"""
        examples = []
        try:
            with open(self.base_data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        except FileNotFoundError:
            print(f"警告: ベースデータファイル {self.base_data_file} が見つかりません")
            return []
        
        print(f"ベースデータを読み込み: {len(examples)}件")
        return examples
    
    def _load_meeting_contexts(self) -> Dict[str, Dict[str, Any]]:
        """会議コンテキストの拡張情報"""
        return {
            "定例会議": {
                "variations": ["週次定例会議", "月次定例会議", "四半期定例会議"],
                "time_prefixes": ["来週の", "今月の", "次回の"],
                "urgency": "通常",
                "formality": "中"
            },
            "プロジェクト会議": {
                "variations": ["プロジェクトキックオフ会議", "プロジェクト進捗会議", "プロジェクト完了会議"],
                "time_prefixes": ["明日の", "来週の", "緊急の"],
                "urgency": "高",
                "formality": "高"
            },
            "企画会議": {
                "variations": ["新規企画会議", "企画レビュー会議", "企画承認会議"],
                "time_prefixes": ["今週の", "来月の", "年度末の"],
                "urgency": "中",
                "formality": "高"
            },
            "部長会議": {
                "variations": ["部長定例会議", "部長緊急会議", "部長戦略会議"],
                "time_prefixes": ["明日の", "今週金曜の", "月末の"],
                "urgency": "高",
                "formality": "最高"
            },
            "戦略会議": {
                "variations": ["経営戦略会議", "事業戦略会議", "マーケティング戦略会議"],
                "time_prefixes": ["来期の", "年度の", "中期計画の"],
                "urgency": "最高",
                "formality": "最高"
            }
        }
    
    def _load_industry_contexts(self) -> Dict[str, Dict[str, Any]]:
        """業界別のコンテキスト情報"""
        return {
            "technology": {
                "name": "IT・テクノロジー",
                "meeting_style": "アジャイル・効率重視",
                "terminology": ["スプリント", "リリース", "デプロイ", "API", "クラウド"],
                "topics": ["システム開発", "プロダクト企画", "技術選定", "インフラ構築"]
            },
            "finance": {
                "name": "金融・銀行",
                "meeting_style": "フォーマル・コンプライアンス重視",
                "terminology": ["ROI", "リスク管理", "ポートフォリオ", "規制対応", "監査"],
                "topics": ["投資戦略", "リスク評価", "コンプライアンス", "業績分析"]
            },
            "healthcare": {
                "name": "ヘルスケア・医療",
                "meeting_style": "エビデンスベース・患者中心",
                "terminology": ["患者アウトカム", "臨床試験", "医療安全", "QOL", "診療ガイドライン"],
                "topics": ["患者ケア改善", "医療安全", "診療プロトコル", "研究計画"]
            },
            "manufacturing": {
                "name": "製造業",
                "meeting_style": "品質重視・継続的改善",
                "terminology": ["カイゼン", "品質管理", "生産性向上", "サプライチェーン", "安全管理"],
                "topics": ["生産計画", "品質改善", "コスト削減", "安全対策"]
            },
            "retail": {
                "name": "小売・流通",
                "meeting_style": "顧客志向・スピード重視",
                "terminology": ["顧客満足度", "在庫管理", "売上分析", "マーケティング", "店舗運営"],
                "topics": ["売上向上", "顧客体験", "在庫最適化", "販促企画"]
            }
        }
    
    def _load_prompt_variations(self) -> Dict[str, List[str]]:
        """プロンプトの拡張パターン"""
        return {
            "prefixes": [
                "",
                "緊急で",
                "至急",
                "来週までに",
                "月末までに",
                "詳細に",
                "簡潔に",
                "包括的に"
            ],
            "contexts": [
                "",
                "クライアント向けの",
                "社内向けの",
                "役員向けの",
                "チーム向けの",
                "外部パートナー向けの"
            ],
            "additional_requirements": [
                "",
                "資料も含めて",
                "予算情報も含めて", 
                "スケジュールも含めて",
                "リスク分析も含めて",
                "成功指標も含めて",
                "前回からの変更点も含めて"
            ]
        }
    
    def enhance_base_example(self, base_example: Dict[str, Any], 
                           enhancement_type: str = "standard") -> Dict[str, Any]:
        """ベース例を拡張して新しい例を生成"""
        enhanced = copy.deepcopy(base_example)
        
        if enhancement_type == "industry_context":
            enhanced = self._add_industry_context(enhanced)
        elif enhancement_type == "complexity_increase":
            enhanced = self._increase_complexity(enhanced)
        elif enhancement_type == "prompt_variation":
            enhanced = self._vary_prompt(enhanced)
        elif enhancement_type == "parameter_variation":
            enhanced = self._vary_parameters(enhanced)
        elif enhancement_type == "combination":
            enhanced = self._combine_instructions(enhanced)
        
        return enhanced
    
    def _add_industry_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """業界コンテキストを追加"""
        industry_key = random.choice(list(self.industry_contexts.keys()))
        industry = self.industry_contexts[industry_key]
        
        # プロンプトに業界コンテキストを追加
        original_prompt = example["prompt"]
        
        # 業界特有の用語を含む新しいプロンプト
        enhanced_prompt = f"{industry['name']}における{original_prompt}"
        
        # 業界特有のトピックを組み込み
        if "について" in enhanced_prompt:
            topic = random.choice(industry["topics"])
            enhanced_prompt = enhanced_prompt.replace("について", f"（{topic}に関連して）について")
        
        example["prompt"] = enhanced_prompt
        
        # メタデータに業界情報を追加
        if "metadata" not in example:
            example["metadata"] = {}
        example["metadata"]["industry"] = industry_key
        example["metadata"]["industry_context"] = industry
        
        return example
    
    def _increase_complexity(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """複雑度を増加"""
        # 既存の指示に追加の指示を組み合わせ
        current_instructions = example["instruction_id_list"]
        available_instructions = list(self.registry.keys())
        
        # 現在の指示と異なるカテゴリから追加
        used_categories = set()
        for inst_id in current_instructions:
            for cat_name, cat_prefix in self.categories.items():
                if inst_id.startswith(cat_prefix):
                    used_categories.add(cat_prefix)
        
        # 未使用カテゴリから選択
        unused_instructions = [
            inst for inst in available_instructions 
            if not any(inst.startswith(cat) for cat in used_categories)
        ]
        
        if unused_instructions:
            additional_instruction = random.choice(unused_instructions)
            example["instruction_id_list"].append(additional_instruction)
            
            # 対応するkwargsを追加
            additional_kwargs = self._generate_kwargs_for_instruction(additional_instruction)
            example["kwargs"].append(additional_kwargs)
            
            # プロンプトを更新
            example["prompt"] = self._update_prompt_for_additional_instruction(
                example["prompt"], additional_instruction
            )
        
        return example
    
    def _vary_prompt(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """プロンプトのバリエーションを生成"""
        original_prompt = example["prompt"]
        
        # プレフィックスを追加
        prefix = random.choice(self.prompt_variations["prefixes"])
        context = random.choice(self.prompt_variations["contexts"])
        additional = random.choice(self.prompt_variations["additional_requirements"])
        
        # 新しいプロンプトを構築
        new_prompt = original_prompt
        
        if prefix:
            new_prompt = f"{prefix}{new_prompt}"
        
        if context:
            new_prompt = context + new_prompt
            
        if additional:
            if "。" in new_prompt:
                new_prompt = new_prompt.replace("。", f"。{additional}")
            else:
                new_prompt = f"{new_prompt}{additional}"
        
        example["prompt"] = new_prompt
        
        return example
    
    def _vary_parameters(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータのバリエーションを生成"""
        for i, kwargs in enumerate(example["kwargs"]):
            instruction_id = example["instruction_id_list"][i]
            
            # 指示に応じてパラメータを変更
            if "agenda_format" in instruction_id:
                if "num_agenda_items" in kwargs:
                    # 項目数を変更（元の値から±2の範囲）
                    original = kwargs["num_agenda_items"]
                    new_value = max(2, original + random.randint(-2, 3))
                    kwargs["num_agenda_items"] = new_value
            
            elif "participants_list" in instruction_id:
                if "min_participants" in kwargs:
                    original = kwargs["min_participants"]
                    new_value = max(2, original + random.randint(-1, 4))
                    kwargs["min_participants"] = new_value
                
                # 部署情報の要求を変更
                kwargs["include_department"] = random.choice([True, False])
            
            elif "action_items" in instruction_id:
                if "min_items" in kwargs:
                    original = kwargs["min_items"]
                    new_value = max(1, original + random.randint(-1, 3))
                    kwargs["min_items"] = new_value
                
                kwargs["include_deadline"] = random.choice([True, False])
            
            elif "time_management" in instruction_id:
                kwargs["include_date"] = random.choice([True, False])
        
        return example
    
    def _combine_instructions(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """他のベース例から指示を組み合わせ"""
        if len(self.base_examples) <= 1:
            return example
        
        # 他のベース例から指示を借用
        other_examples = [ex for ex in self.base_examples if ex["key"] != example["key"]]
        other_example = random.choice(other_examples)
        
        # 重複しない指示を追加
        for i, other_inst in enumerate(other_example["instruction_id_list"]):
            if other_inst not in example["instruction_id_list"]:
                example["instruction_id_list"].append(other_inst)
                
                # 対応するkwargsを追加
                if i < len(other_example["kwargs"]):
                    example["kwargs"].append(copy.deepcopy(other_example["kwargs"][i]))
                else:
                    example["kwargs"].append({})
                
                # 最初の1つだけ追加
                break
        
        # プロンプトを統合
        example["prompt"] = self._merge_prompts(example["prompt"], other_example["prompt"])
        
        return example
    
    def _generate_kwargs_for_instruction(self, instruction_id: str) -> Dict[str, Any]:
        """指示IDに基づいてkwargsを生成"""
        kwargs = {}
        
        if "agenda_format" in instruction_id:
            kwargs["num_agenda_items"] = random.randint(3, 8)
        elif "participants_list" in instruction_id:
            kwargs["min_participants"] = random.randint(3, 10)
            kwargs["include_department"] = random.choice([True, False])
        elif "action_items" in instruction_id:
            kwargs["min_items"] = random.randint(2, 6)
            kwargs["include_deadline"] = random.choice([True, False])
        elif "time_management" in instruction_id:
            kwargs["include_date"] = random.choice([True, False])
        elif "location_setting" in instruction_id:
            kwargs["location_type"] = random.choice(["room", "online", "hybrid"])
        
        return kwargs
    
    def _update_prompt_for_additional_instruction(self, original_prompt: str, instruction_id: str) -> str:
        """追加指示に基づいてプロンプトを更新"""
        additions = {
            "location_setting": "会議の場所情報も含めて",
            "time_management": "時間の詳細も含めて",
            "action_items": "アクションアイテムも含めて",
            "decisions_format": "決定事項も含めて",
            "summary_format": "要約も含めて"
        }
        
        for key, addition in additions.items():
            if key in instruction_id:
                if "。" in original_prompt:
                    return original_prompt.replace("。", f"。{addition}")
                else:
                    return f"{original_prompt}{addition}"
        
        return original_prompt
    
    def _merge_prompts(self, prompt1: str, prompt2: str) -> str:
        """2つのプロンプトを自然に結合"""
        # シンプルな結合ロジック
        base1 = prompt1.replace("。", "").replace("してください", "")
        base2 = prompt2.replace("。", "").replace("してください", "")
        
        # 重複する単語を避けて結合
        if "会議" in base1 and "会議" in base2:
            return f"{base1}と{base2.replace('会議', '')}してください。"
        else:
            return f"{base1}と{base2}してください。"
    
    def generate_enhanced_dataset(self, 
                                num_examples: int = 100,
                                enhancement_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """ベースデータを基に拡張データセットを生成"""
        
        if not self.base_examples:
            print("エラー: ベースデータが見つかりません")
            return []
        
        if enhancement_distribution is None:
            enhancement_distribution = {
                "standard": 0.2,
                "industry_context": 0.2,
                "complexity_increase": 0.2,
                "prompt_variation": 0.2,
                "parameter_variation": 0.1,
                "combination": 0.1
            }
        
        enhanced_examples = []
        enhancement_types = list(enhancement_distribution.keys())
        weights = list(enhancement_distribution.values())
        
        for i in range(num_examples):
            # ベース例をランダムに選択
            base_example = random.choice(self.base_examples)
            
            # 拡張タイプを選択
            enhancement_type = random.choices(enhancement_types, weights=weights)[0]
            
            # 拡張実行
            enhanced = self.enhance_base_example(base_example, enhancement_type)
            
            # 新しいキーを割り当て
            enhanced["key"] = i + 1
            
            # 拡張情報をメタデータに記録
            if "metadata" not in enhanced:
                enhanced["metadata"] = {}
            enhanced["metadata"]["base_key"] = base_example["key"]
            enhanced["metadata"]["enhancement_type"] = enhancement_type
            enhanced["metadata"]["generation_timestamp"] = datetime.now().isoformat()
            
            enhanced_examples.append(enhanced)
        
        return enhanced_examples
    
    def save_enhanced_data(self, examples: List[Dict[str, Any]], output_path: str):
        """拡張されたデータを保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        print(f"拡張データ生成完了: {len(examples)}件を{output_path}に保存")
    
    def generate_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """拡張データの統計情報を生成"""
        stats = {
            "total_examples": len(examples),
            "base_data_examples": len(self.base_examples),
            "enhancement_distribution": defaultdict(int),
            "instruction_distribution": defaultdict(int),
            "complexity_distribution": defaultdict(int),
            "industry_distribution": defaultdict(int)
        }
        
        for example in examples:
            # 拡張タイプの分布
            if "metadata" in example and "enhancement_type" in example["metadata"]:
                stats["enhancement_distribution"][example["metadata"]["enhancement_type"]] += 1
            
            # 指示の複雑度分布
            num_instructions = len(example["instruction_id_list"])
            stats["complexity_distribution"][f"{num_instructions}_instructions"] += 1
            
            # 指示分布
            for inst_id in example["instruction_id_list"]:
                stats["instruction_distribution"][inst_id] += 1
            
            # 業界分布
            if "metadata" in example and "industry" in example["metadata"]:
                stats["industry_distribution"][example["metadata"]["industry"]] += 1
        
        return dict(stats)

def main():
    parser = argparse.ArgumentParser(description="jMeet-Eval拡張データ生成（ベースデータ使用）")
    parser.add_argument("--base_data", type=str, 
                       default="../data/jmeet_input_data.jsonl",
                       help="ベースとなる入力データファイル")
    parser.add_argument("--num_examples", type=int, default=50, 
                       help="生成する例の数")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    parser.add_argument("--output", type=str, 
                       default="../data/enhanced_jmeet_input_data.jsonl",
                       help="出力ファイルパス")
    parser.add_argument("--stats", action="store_true", help="統計情報を表示")
    
    args = parser.parse_args()
    
    # ジェネレータを初期化
    print("ベースデータを使用した拡張データセット生成中...")
    generator = AdvancedMeetingDataGenerator(
        base_data_file=args.base_data, 
        seed=args.seed
    )
    
    # 拡張データ生成
    enhanced_examples = generator.generate_enhanced_dataset(
        num_examples=args.num_examples
    )
    
    if not enhanced_examples:
        print("エラー: データ生成に失敗しました")
        return
    
    # ファイルに保存
    generator.save_enhanced_data(enhanced_examples, args.output)
    
    # 統計情報を表示
    if args.stats:
        stats = generator.generate_statistics(enhanced_examples)
        print("\n=== 拡張データ統計 ===")
        print(f"ベースデータ: {stats['base_data_examples']}件")
        print(f"生成データ: {stats['total_examples']}件")
        
        print("\n拡張タイプ分布:")
        for enhancement_type, count in stats["enhancement_distribution"].items():
            percentage = count / stats["total_examples"] * 100
            print(f"  {enhancement_type}: {count} ({percentage:.1f}%)")
        
        print("\n複雑度分布:")
        for complexity, count in sorted(stats["complexity_distribution"].items()):
            percentage = count / stats["total_examples"] * 100
            print(f"  {complexity}: {count} ({percentage:.1f}%)")
        
        if stats["industry_distribution"]:
            print("\n業界分布:")
            for industry, count in stats["industry_distribution"].items():
                percentage = count / stats["total_examples"] * 100
                print(f"  {industry}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()