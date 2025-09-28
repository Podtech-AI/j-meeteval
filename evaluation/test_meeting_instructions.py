# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 包括的テストスイート"""

import sys
import os
import unittest
import time

# 親ディレクトリをパスに追加（jmeet-eval構造用）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meeting_instructions import *
import meeting_instructions_registry
import meeting_evaluation_lib

class TestMeetingInstructions(unittest.TestCase):
    """会議指示クラスの包括的テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.evaluator = meeting_evaluation_lib.MeetingEvaluator(strict=True, verbose=False)
    
    def test_filler_removal_checker(self):
        """フィラー除去チェッカーのテスト"""
        checker = FillerRemovalChecker("test")
        
        # 成功例（フィラーなし）
        clean_text = "今日の会議では新しいマーケティング施策について話し合いました。"
        self.assertTrue(checker.check_following(clean_text))
        
        # 失敗例（フィラーあり）
        filler_text = "あのー、今日の会議では、えっと、新しいマーケティング施策について話し合いました。"
        self.assertFalse(checker.check_following(filler_text))
    
    def test_no_ending_phrase_checker(self):
        """末尾句除去チェッカーのテスト"""
        checker = NoEndingPhraseChecker("test")
        
        # 成功例（「以上」なし）
        clean_text = "会議の内容をまとめました。"
        self.assertTrue(checker.check_following(clean_text))
        
        # 失敗例（「以上」あり）
        ending_text = "会議の内容をまとめました。以上です。"
        self.assertFalse(checker.check_following(ending_text))
    
    def test_no_punctuation_checker(self):
        """句点除去チェッカーのテスト"""
        checker = NoPunctuationChecker("test")
        
        # 成功例（句点なし）
        no_punct = "会議の内容をまとめました"
        self.assertTrue(checker.check_following(no_punct))
        
        # 失敗例（句点あり）
        with_punct = "会議の内容をまとめました。"
        self.assertFalse(checker.check_following(with_punct))
    
    def test_character_limit_checker(self):
        """文字数制限チェッカーのテスト"""
        checker = CharacterLimitChecker("test")
        
        # 短いテキスト（成功）
        short_text = "短いテキスト"
        checker._cached_kwargs = {"max_chars": 100}
        self.assertTrue(checker.check_following(short_text))
        
        # 長いテキスト（失敗）
        long_text = "非常に長いテキスト" * 50
        self.assertFalse(checker.check_following(long_text))
    
    def test_bullet_points_checker(self):
        """箇条書きチェッカーのテスト"""
        checker = BulletPointsChecker("test")
        
        # 5点の箇条書き（成功）
        bullet_text = """
        ・ポイント1
        ・ポイント2
        ・ポイント3
        ・ポイント4
        ・ポイント5
        """
        checker._cached_kwargs = {"num_points": 5, "one_line": True}
        self.assertTrue(checker.check_following(bullet_text))
    
    def test_structured_sections_checker(self):
        """構造化セクションチェッカーのテスト"""
        checker = StructuredSectionsChecker("test")
        
        # 指定セクション順（成功）
        sectioned_text = """
        ## 概要
        会議の概要です。
        
        ## 議論
        議論の内容です。
        
        ## 決定事項
        決定した事項です。
        """
        checker._cached_kwargs = {"sections": ["概要", "議論", "決定事項"], "use_markdown": True}
        self.assertTrue(checker.check_following(sectioned_text))
    
    def test_json_format_checker(self):
        """JSONフォーマットチェッカーのテスト"""
        checker = JsonFormatChecker("test")
        
        # 有効なJSON（成功）
        valid_json = '{"summary": "会議要約", "items": [{"topic": "議題1", "points": ["ポイント1"]}]}'
        checker._cached_kwargs = {"json_schema": "summary_items"}
        self.assertTrue(checker.check_following(valid_json))
    
    def test_writing_style_checker(self):
        """文体チェッカーのテスト"""
        checker = WritingStyleChecker("test")
        
        # です・ます調（成功）
        desu_masu_text = "会議の内容をまとめました。次回も開催します。"
        checker._cached_kwargs = {"style_type": "desu_masu"}
        self.assertTrue(checker.check_following(desu_masu_text))
    
    def test_markdown_table_checker(self):
        """マークダウン表チェッカーのテスト"""
        checker = MarkdownTableChecker("test")
        
        # 有効な表（成功）
        table_text = """
        | 論点 | 結論 | 担当 | 期限 |
        |------|------|------|------|
        | 議題1 | 決定1 | 田中 | 月末 |
        """
        checker._cached_kwargs = {"columns": ["論点", "結論", "担当", "期限"]}
        self.assertTrue(checker.check_following(table_text))
    
    def test_evaluation_grade_checker(self):
        """評価グレードチェッカーのテスト"""
        checker = EvaluationGradeChecker("test")
        
        # A評価（成功）
        grade_text = "会議の要約です。評価：A"
        checker._cached_kwargs = {"grade_format": "ABC"}
        self.assertTrue(checker.check_following(grade_text))

class TestMeetingEvaluationLib(unittest.TestCase):
    """評価ライブラリのテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.evaluator = meeting_evaluation_lib.MeetingEvaluator(strict=True, verbose=False)
    
    def test_single_instruction_evaluation(self):
        """単一指示評価のテスト"""
        instruction_id = "output_constraint:filler_removal"
        prompt = "あのー、会議の内容です"
        response = "会議の内容です"
        kwargs = {}
        
        # 成功応答
        result, error_detail = self.evaluator.evaluate_single_instruction(
            instruction_id, prompt, response, kwargs
        )
        self.assertTrue(result)
    
    def test_response_evaluation(self):
        """応答評価のテスト"""
        # MeetingInputExampleオブジェクトを作成
        input_example = meeting_evaluation_lib.MeetingInputExample(
            key=1,
            instruction_id_list=["output_constraint:filler_removal"],
            prompt="あのー、会議の内容です",
            kwargs=[{}]
        )
        
        # 成功応答
        response = "会議の内容です"
        result = self.evaluator.evaluate_response(input_example, response)
        self.assertTrue(result.follow_all_instructions)
    
    def test_metrics_calculation(self):
        """メトリクス計算のテスト"""
        outputs = [
            meeting_evaluation_lib.MeetingOutputExample(
                key=1,
                instruction_id_list=["output_constraint:filler_removal"],
                prompt="テストプロンプト1",
                response="テスト応答1",
                follow_all_instructions=True,
                follow_instruction_list=[True],
                execution_time=0.1
            ),
            meeting_evaluation_lib.MeetingOutputExample(
                key=2,
                instruction_id_list=["output_constraint:no_punctuation"],
                prompt="テストプロンプト2",
                response="テスト応答2",
                follow_all_instructions=False,
                follow_instruction_list=[False],
                execution_time=0.1
            )
        ]
        
        metrics = meeting_evaluation_lib.calculate_meeting_metrics(outputs)
        self.assertEqual(metrics["prompt_level_strict_accuracy"], 0.5)
        self.assertEqual(metrics["instruction_level_strict_accuracy"], 0.5)

class TestMeetingRegistry(unittest.TestCase):
    """レジストリのテスト"""
    
    def test_registry_completeness(self):
        """レジストリの完全性テスト"""
        registry = meeting_instructions_registry.MEETING_INSTRUCTION_DICT
        
        # 期待される指示数（10カテゴリの合計）
        expected_count = 22  # 実際の指示数に合わせて調整
        actual_count = len(registry)
        
        self.assertEqual(actual_count, expected_count,
                        f"Expected {expected_count} instructions, got {actual_count}")
    
    def test_instruction_info(self):
        """指示情報取得のテスト"""
        instruction_id = "output_constraint:filler_removal"
        info = meeting_instructions_registry.get_instruction_info(instruction_id)
        
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], instruction_id)
        self.assertEqual(info["category"], "output_constraint")
    
    def test_category_distribution(self):
        """カテゴリ分布のテスト"""
        stats = meeting_instructions_registry.get_registry_statistics()
        
        # 基本カテゴリが存在することを確認
        expected_categories = [
            "output_constraint", "language_translation", "length_control",
            "annotation", "template_filling", "formatting",
            "content_evaluation", "json_output", "writing_style", "table_format"
        ]
        
        for category in expected_categories:
            self.assertIn(category, stats["categories"])
    
    def test_instruction_examples(self):
        """指示例生成のテスト"""
        examples = meeting_instructions_registry.generate_instruction_examples()
        
        # 各指示に例が生成されていることを確認
        for instruction_id in meeting_instructions_registry.AVAILABLE_INSTRUCTION_IDS:
            self.assertIn(instruction_id, examples)
            self.assertIn("default", examples[instruction_id])

class TestPerformance(unittest.TestCase):
    """パフォーマンステスト"""
    
    def test_evaluation_speed(self):
        """評価速度のテスト"""
        evaluator = meeting_evaluation_lib.MeetingEvaluator(strict=True, verbose=False)
        
        # 簡単な評価例
        example = {
            "instruction_id_list": ["output_constraint:filler_removal"],
            "kwargs": [{}]
        }
        
        response = "清潔な応答"
        
        start_time = time.time()
        for _ in range(100):
            evaluator.evaluate_single_instruction(
                "output_constraint:filler_removal",
                "テストプロンプト",
                response,
                {}
            )
        end_time = time.time()
        
        # 100回の評価が1秒以内（目安）
        self.assertLess(end_time - start_time, 1.0)

if __name__ == '__main__':
    # 詳細なテストログ出力
    unittest.main(verbosity=2)