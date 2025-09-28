# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 会議関連指示のレジストリ"""

import meeting_instructions as instructions

# 必要な指示のみインポート
from meeting_instructions import (
    # 出力制約関連
    FillerRemovalChecker,
    NoEndingPhraseChecker,
    NoPunctuationChecker,
    OnlyMeetingMinutesChecker,
    
    # 言語指定・翻訳関連
    JapaneseSummaryChecker,
    EnglishTranslationChecker,
    
    # 分量指定関連
    CharacterLimitChecker,
    ProportionChecker,
    SentenceLimitChecker,
    
    # 付記関連
    EndingNoteChecker,
    BeginningNoteChecker,
    
    # テンプレート充填関連
    TemplateFillingChecker,
    
    # 箇条書き関連
    BulletPointsChecker,
    
    # 章立て・セクション関連
    StructuredSectionsChecker,
    QAFormatChecker,
    RequiredItemsChecker,
    
    # 内容評価関連
    EvaluationGradeChecker,
    FivePointScaleChecker,
    
    # JSON出力関連
    JsonFormatChecker,
    
    # 文体・敬語関連
    WritingStyleChecker,
    NumberFormatChecker,
    
    # 表形式関連
    MarkdownTableChecker
)

# 会議関連指示のカテゴリ定義
_OUTPUT_CONSTRAINT = "output_constraint:"
_LANGUAGE_TRANSLATION = "language_translation:"
_LENGTH_CONTROL = "length_control:"
_ANNOTATION = "annotation:"
_TEMPLATE_FILLING = "template_filling:"
_FORMATTING = "formatting:"
_CONTENT_EVALUATION = "content_evaluation:"
_JSON_OUTPUT = "json_output:"
_WRITING_STYLE = "writing_style:"
_TABLE_FORMAT = "table_format:"

# 会議関連指示のレジストリ
MEETING_INSTRUCTION_DICT = {
    # 出力制約関連（Output Constraints）
    _OUTPUT_CONSTRAINT + "filler_removal": FillerRemovalChecker,
    _OUTPUT_CONSTRAINT + "no_ending_phrase": NoEndingPhraseChecker,
    _OUTPUT_CONSTRAINT + "no_punctuation": NoPunctuationChecker,
    _OUTPUT_CONSTRAINT + "only_minutes": OnlyMeetingMinutesChecker,
    
    # 言語指定・翻訳関連（Language and Translation）
    _LANGUAGE_TRANSLATION + "japanese_summary": JapaneseSummaryChecker,
    _LANGUAGE_TRANSLATION + "english_translation": EnglishTranslationChecker,
    
    # 分量指定関連（Length Control）
    _LENGTH_CONTROL + "character_limit": CharacterLimitChecker,
    _LENGTH_CONTROL + "proportion": ProportionChecker,
    _LENGTH_CONTROL + "sentence_limit": SentenceLimitChecker,
    
    # 付記関連（Annotations）
    _ANNOTATION + "ending_note": EndingNoteChecker,
    _ANNOTATION + "beginning_note": BeginningNoteChecker,
    
    # テンプレート充填関連（Template Filling）
    _TEMPLATE_FILLING + "template": TemplateFillingChecker,
    
    # 箇条書き・フォーマット関連（Formatting）
    _FORMATTING + "bullet_points": BulletPointsChecker,
    _FORMATTING + "structured_sections": StructuredSectionsChecker,
    _FORMATTING + "qa_format": QAFormatChecker,
    _FORMATTING + "required_items": RequiredItemsChecker,
    
    # 内容評価関連（Content Evaluation）
    _CONTENT_EVALUATION + "evaluation_grade": EvaluationGradeChecker,
    _CONTENT_EVALUATION + "five_point_scale": FivePointScaleChecker,
    
    # JSON出力関連（JSON Output）
    _JSON_OUTPUT + "json_format": JsonFormatChecker,
    
    # 文体・敬語関連（Writing Style）
    _WRITING_STYLE + "writing_style": WritingStyleChecker,
    _WRITING_STYLE + "number_format": NumberFormatChecker,
    
    # 表形式関連（Table Format）
    _TABLE_FORMAT + "markdown_table": MarkdownTableChecker
}

# 利用可能な指示IDのリスト
AVAILABLE_INSTRUCTION_IDS = list(MEETING_INSTRUCTION_DICT.keys())

# カテゴリ別の指示IDを取得する関数
def get_instructions_by_category(category_prefix: str) -> list:
    """指定されたカテゴリプレフィックスに一致する指示IDを返す"""
    return [instruction_id for instruction_id in AVAILABLE_INSTRUCTION_IDS 
            if instruction_id.startswith(category_prefix)]

# 全カテゴリの定義
MEETING_CATEGORIES = {
    "output_constraint": _OUTPUT_CONSTRAINT,
    "language_translation": _LANGUAGE_TRANSLATION,
    "length_control": _LENGTH_CONTROL,
    "annotation": _ANNOTATION,
    "template_filling": _TEMPLATE_FILLING,
    "formatting": _FORMATTING,
    "content_evaluation": _CONTENT_EVALUATION,
    "json_output": _JSON_OUTPUT,
    "writing_style": _WRITING_STYLE,
    "table_format": _TABLE_FORMAT
}

# 指示の詳細情報を取得する関数
def get_instruction_info(instruction_id: str) -> dict:
    """指示IDから詳細情報を取得"""
    if instruction_id not in MEETING_INSTRUCTION_DICT:
        return None
    
    instruction_class = MEETING_INSTRUCTION_DICT[instruction_id]
    instance = instruction_class(instruction_id)
    
    # カテゴリを特定
    category = None
    for cat_name, cat_prefix in MEETING_CATEGORIES.items():
        if instruction_id.startswith(cat_prefix):
            category = cat_name
            break
    
    return {
        "id": instruction_id,
        "category": category,
        "class_name": instruction_class.__name__,
        "args": instance.get_instruction_args() if hasattr(instance, 'get_instruction_args') else {},
        "description": instance.__doc__
    }

# 全指示の統計情報を取得
def get_registry_statistics() -> dict:
    """レジストリの統計情報を返す"""
    stats = {
        "total_instructions": len(AVAILABLE_INSTRUCTION_IDS),
        "categories": {}
    }
    
    for cat_name, cat_prefix in MEETING_CATEGORIES.items():
        instructions = get_instructions_by_category(cat_prefix)
        stats["categories"][cat_name] = {
            "count": len(instructions),
            "instructions": instructions
        }
    
    return stats

# 指示のサンプルを生成する関数
def generate_instruction_examples() -> dict:
    """各指示のサンプル例を生成"""
    examples = {}
    
    for inst_id, inst_class in MEETING_INSTRUCTION_DICT.items():
        instance = inst_class(inst_id)
        
        # デフォルト引数での例
        default_example = instance.generate_instruction()
        
        # カスタム引数での例（可能な場合）
        custom_examples = []
        if hasattr(instance, 'get_instruction_args'):
            args_info = instance.get_instruction_args()
            if args_info:
                # 最小値での例
                min_args = {k: v.get('min', v.get('default', 1)) 
                          for k, v in args_info.items() if isinstance(v, dict)}
                if min_args:
                    custom_examples.append({
                        "args": min_args,
                        "instruction": instance.generate_instruction(**min_args)
                    })
                
                # 最大値での例
                max_args = {k: v.get('max', v.get('default', 5)) 
                          for k, v in args_info.items() if isinstance(v, dict)}
                if max_args and max_args != min_args:
                    custom_examples.append({
                        "args": max_args,
                        "instruction": instance.generate_instruction(**max_args)
                    })
        
        examples[inst_id] = {
            "default": default_example,
            "custom": custom_examples
        }
    
    return examples