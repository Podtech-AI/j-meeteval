# coding=utf-8
# Copyright 2025 jMeet-Eval Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""jMeet-Eval: 会議関連指示の評価ライブラリ"""

import re
import json
from typing import Dict, Optional, Union, List, Any
from abc import ABC, abstractmethod
from datetime import datetime
import random

class MeetingInstruction(ABC):
    """会議関連指示の基底抽象クラス"""
    
    def __init__(self, instruction_id: str):
        self.instruction_id = instruction_id
        self._cached_kwargs = None
    
    @abstractmethod
    def generate_instruction(self, **kwargs) -> str:
        """指示文を生成する抽象メソッド"""
        pass
    
    @abstractmethod
    def check_following(self, value: str) -> bool:
        """指示に従っているかチェックする抽象メソッド"""
        pass
    
    def get_instruction_args(self) -> Dict[str, Any]:
        """この指示で使用可能な引数を返す"""
        return {}
    
    def build_description(self, **kwargs) -> str:
        """指示の詳細説明を構築"""
        return self.generate_instruction(**kwargs)




# === 出力制約関連の指示 ===

class FillerRemovalChecker(MeetingInstruction):
    """フィラーの除去をチェック"""
    
    def generate_instruction(self, **kwargs) -> str:
        return "あのー、えっと、などのフィラーを除いてください。"
    
    def check_following(self, value: str) -> bool:
        # よくあるフィラーのリスト
        fillers = [
            "あのー", "あの～", "あのう", "あのぉ",
            "えっと", "えーっと", "ええと", "えーと", "えと",
            "その～", "そのー", "そのう",
            "まあ", "まぁ", 
            "なんか", "なんていうか",
            "ええ", "えー", "ええー",
            "うーん", "うん", "ううん",
            "あー", "ああ", "あ～"
        ]
        
        # 各フィラーが含まれていないかチェック
        for filler in fillers:
            if filler in value:
                return False
        return True

class NoEndingPhraseChecker(MeetingInstruction):
    """末尾の「以上」が不要であることをチェック"""
    
    def generate_instruction(self, **kwargs) -> str:
        return "末尾の「以上」は不要です。"
    
    def check_following(self, value: str) -> bool:
        # 文末の「以上」をチェック
        value_stripped = value.strip()
        endings = ["以上", "以上です", "以上。", "以上です。"]
        
        for ending in endings:
            if value_stripped.endswith(ending):
                return False
        return True

class NoPunctuationChecker(MeetingInstruction):
    """句点の除去をチェック"""
    
    def generate_instruction(self, **kwargs) -> str:
        return "句点は除くこと。"
    
    def check_following(self, value: str) -> bool:
        # 句点（。）が含まれていないかチェック
        return "。" not in value

class OnlyMeetingMinutesChecker(MeetingInstruction):
    """議事録以外の出力がないことをチェック"""
    
    def generate_instruction(self, **kwargs) -> str:
        return "議事録以外は出力しないこと。"
    
    def check_following(self, value: str) -> bool:
        # 議事録に関連しない余計な説明文が含まれていないかチェック
        unnecessary_phrases = [
            "私は", "私が", "私の意見では",
            "説明します", "解説します", "紹介します",
            "以下が議事録です", "議事録を作成しました",
            "ご確認ください", "よろしくお願いします",
            "参考までに", "補足ですが"
        ]
        
        for phrase in unnecessary_phrases:
            if phrase in value:
                return False
        return True

# === 言語指定・翻訳関連の指示 ===

class JapaneseSummaryChecker(MeetingInstruction):
    """要約が日本語であることをチェック"""
    
    def generate_instruction(self, **kwargs) -> str:
        return "要約は日本語で記載してください。"
    
    def check_following(self, value: str) -> bool:
        # langdetectを使用して言語を検出（instructions.pyを参考）
        try:
            # langdetect実行時インポート（オプション依存）
            import langdetect
            # 要約部分を抽出（要約、サマリー、概要などのキーワードの後の文）
            summary_keywords = ["要約", "サマリー", "概要", "まとめ"]
            
            for keyword in summary_keywords:
                if keyword in value:
                    # キーワード以降のテキストを取得
                    summary_start = value.find(keyword)
                    summary_text = value[summary_start:]
                    
                    # 言語検出
                    detected_lang = langdetect.detect(summary_text[:200])  # 最初の200文字で判定
                    if detected_lang != "ja":
                        return False
            
            return True
        except:
            # langdetectが使えない場合は、日本語文字の存在でチェック
            import re
            japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'
            return bool(re.search(japanese_pattern, value))

class EnglishTranslationChecker(MeetingInstruction):
    """英訳の指示に従っているかチェック"""
    
    def generate_instruction(self, translate_what: str = "全体", **kwargs) -> str:
        return f"{translate_what}を英訳してください。"
    
    def check_following(self, value: str) -> bool:
        # 英語と日本語の両方が含まれているかチェック
        try:
            # langdetect実行時インポート（オプション依存）
            import langdetect
            # 複数言語が含まれているか検出
            langs = langdetect.detect_langs(value)
            lang_codes = [lang.lang for lang in langs]
            
            # 英語と日本語の両方が含まれているか
            return 'en' in lang_codes and 'ja' in lang_codes
        except:
            # langdetectが使えない場合は、英語文字の存在でチェック
            import re
            english_pattern = r'[a-zA-Z]{3,}'  # 3文字以上の英単語
            japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'
            
            has_english = bool(re.search(english_pattern, value))
            has_japanese = bool(re.search(japanese_pattern, value))
            
            return has_english and has_japanese
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "translate_what": {
                "type": "str",
                "default": "全体",
                "description": "翻訳対象（全体、要約部分、決定事項など）"
            }
        }

# === 分量指定関連の指示 ===

class CharacterLimitChecker(MeetingInstruction):
    """文字数制限をチェック"""
    
    def generate_instruction(self, max_chars: int = 500, exact_range: bool = False, min_chars: int = None, **kwargs) -> str:
        if exact_range and min_chars:
            return f"{min_chars}〜{max_chars}字で記載してください。"
        else:
            return f"{max_chars}文字以内で記載してください。"
    
    def check_following(self, value: str) -> bool:
        # 改行や空白を除いた文字数をカウント
        char_count = len(value.replace('\n', '').replace(' ', '').replace('　', ''))
        
        max_chars = self._cached_kwargs.get('max_chars', 500) if self._cached_kwargs else 500
        exact_range = self._cached_kwargs.get('exact_range', False) if self._cached_kwargs else False
        min_chars = self._cached_kwargs.get('min_chars', None) if self._cached_kwargs else None
        
        if exact_range and min_chars:
            return min_chars <= char_count <= max_chars
        else:
            return char_count <= max_chars
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "max_chars": {
                "type": "int",
                "default": 500,
                "min": 50,
                "max": 2000,
                "description": "最大文字数"
            },
            "exact_range": {
                "type": "bool",
                "default": False,
                "description": "範囲指定かどうか"
            },
            "min_chars": {
                "type": "int",
                "default": None,
                "description": "最小文字数（範囲指定の場合）"
            }
        }

class ProportionChecker(MeetingInstruction):
    """元の文字起こしに対する割合をチェック"""
    
    def generate_instruction(self, proportion: float = 0.5, **kwargs) -> str:
        percentage = int(proportion * 100)
        return f"元の文字起こしの{percentage}割の分量でまとめてください。"
    
    def check_following(self, value: str) -> bool:
        # この実装では元のテキストの長さが不明なため、
        # 実際の使用時は元テキストを別途渡す必要がある
        # ここでは簡易的に実装
        return True  # 実際の実装では元テキストとの比較が必要
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "proportion": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 0.9,
                "description": "元の文字起こしに対する割合"
            }
        }

class SentenceLimitChecker(MeetingInstruction):
    """文数制限をチェック"""
    
    def generate_instruction(self, max_sentences: int = 3, sentence_length: int = None, **kwargs) -> str:
        base = f"{max_sentences}文以内で記載してください。"
        if sentence_length:
            base += f" 1文は{sentence_length}文字以内にしてください。"
        return base
    
    def check_following(self, value: str) -> bool:
        # 文の分割（。！？での分割）
        import re
        sentences = re.split(r'[。！？]', value)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        max_sentences = self._cached_kwargs.get('max_sentences', 3) if self._cached_kwargs else 3
        sentence_length = self._cached_kwargs.get('sentence_length', None) if self._cached_kwargs else None
        
        # 文数チェック
        if len(sentences) > max_sentences:
            return False
        
        # 各文の長さチェック
        if sentence_length:
            for sentence in sentences:
                if len(sentence) > sentence_length:
                    return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "max_sentences": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 20,
                "description": "最大文数"
            },
            "sentence_length": {
                "type": "int",
                "default": None,
                "description": "1文の最大文字数"
            }
        }

# === 付記関連の指示 ===

class EndingNoteChecker(MeetingInstruction):
    """末尾の固定文言をチェック"""
    
    def generate_instruction(self, ending_text: str = "P.S. この要約は内部共有のみです。", **kwargs) -> str:
        return f"末尾に「{ending_text}」と付け加えてください。"
    
    def check_following(self, value: str) -> bool:
        ending_text = self._cached_kwargs.get('ending_text', 'P.S. この要約は内部共有のみです。') if self._cached_kwargs else 'P.S. この要約は内部共有のみです。'
        return value.strip().endswith(ending_text)
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "ending_text": {
                "type": "str",
                "default": "P.S. この要約は内部共有のみです。",
                "description": "末尾に追加する文言"
            }
        }

class BeginningNoteChecker(MeetingInstruction):
    """冒頭の固定文言をチェック"""
    
    def generate_instruction(self, beginning_text: str = "【社外秘】", **kwargs) -> str:
        return f"冒頭に{beginning_text}と明記してください。"
    
    def check_following(self, value: str) -> bool:
        beginning_text = self._cached_kwargs.get('beginning_text', '【社外秘】') if self._cached_kwargs else '【社外秘】'
        return value.strip().startswith(beginning_text)
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "beginning_text": {
                "type": "str",
                "default": "【社外秘】",
                "description": "冒頭に追加する文言"
            }
        }

# === テンプレート充填関連の指示 ===

class TemplateFillingChecker(MeetingInstruction):
    """テンプレート充填の形式をチェック"""
    
    def generate_instruction(self, template_type: str = "basic", **kwargs) -> str:
        if template_type == "basic":
            return "【会議名】[テキスト] 【日時】[YYYY-MM-DD] 【参加者】[カンマ区切り] 【概要】[テキスト]の形式で記載してください。"
        elif template_type == "todo":
            return "■ ToDo ・①：[タスク内容]：対応者[ ]：期限[ ] ・②：[タスク内容]：対応者[ ]：期限[ ]の形式で記載してください。"
        else:
            return "指定されたテンプレートに従って記載してください。"
    
    def check_following(self, value: str) -> bool:
        template_type = self._cached_kwargs.get('template_type', 'basic') if self._cached_kwargs else 'basic'
        
        if template_type == "basic":
            # 必須項目が含まれているかチェック
            required_fields = ["【会議名】", "【日時】", "【参加者】", "【概要】"]
            for field in required_fields:
                if field not in value:
                    return False
            
            # 日付形式のチェック
            date_pattern = r'【日時】\d{4}-\d{2}-\d{2}'
            if not re.search(date_pattern, value):
                return False
                
        elif template_type == "todo":
            # ToDoリストの形式チェック
            if "■ ToDo" not in value:
                return False
            
            # タスク項目の形式チェック
            todo_pattern = r'・\d+：.+：対応者\[.*?\]：期限\[.*?\]'
            matches = re.findall(todo_pattern, value)
            if len(matches) < 1:
                return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "template_type": {
                "type": "str",
                "default": "basic",
                "choices": ["basic", "todo", "custom"],
                "description": "テンプレートの種類"
            }
        }

# === 箇条書き関連の指示 ===

class BulletPointsChecker(MeetingInstruction):
    """箇条書きの形式をチェック"""
    
    def generate_instruction(self, num_points: int = 5, range_points: bool = False, min_points: int = 2, max_points: int = 6, one_line: bool = True, **kwargs) -> str:
        if range_points:
            base = f"箇条書きで{min_points}〜{max_points}点にまとめてください。"
        else:
            base = f"箇条書きで{num_points}点にまとめてください。"
        
        if one_line:
            base += " 箇条書きの各項目は1行で簡潔にしてください。"
        
        return base
    
    def check_following(self, value: str) -> bool:
        # 箇条書きのパターン（・、-、*、数字）
        bullet_patterns = [
            r'^\s*[・●◆■]\s*.+',
            r'^\s*[-\*]\s*.+',
            r'^\s*\d+[.)\.]\s*.+'
        ]
        
        bullet_points = []
        lines = value.split('\n')
        
        for line in lines:
            for pattern in bullet_patterns:
                if re.match(pattern, line.strip()):
                    bullet_points.append(line.strip())
                    break
        
        num_bullets = len(bullet_points)
        
        # 点数チェック
        range_points = self._cached_kwargs.get('range_points', False) if self._cached_kwargs else False
        if range_points:
            min_points = self._cached_kwargs.get('min_points', 2) if self._cached_kwargs else 2
            max_points = self._cached_kwargs.get('max_points', 6) if self._cached_kwargs else 6
            if not (min_points <= num_bullets <= max_points):
                return False
        else:
            expected_points = self._cached_kwargs.get('num_points', 5) if self._cached_kwargs else 5
            if num_bullets != expected_points:
                return False
        
        # 1行制約のチェック
        one_line = self._cached_kwargs.get('one_line', True) if self._cached_kwargs else True
        if one_line:
            for point in bullet_points:
                # 改行が含まれていないかチェック（項目内での改行）
                if '\n' in point:
                    return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "num_points": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 10,
                "description": "箇条書きの点数"
            },
            "range_points": {
                "type": "bool",
                "default": False,
                "description": "範囲指定かどうか"
            },
            "min_points": {
                "type": "int",
                "default": 2,
                "description": "最小点数（範囲指定の場合）"
            },
            "max_points": {
                "type": "int",
                "default": 6,
                "description": "最大点数（範囲指定の場合）"
            },
            "one_line": {
                "type": "bool",
                "default": True,
                "description": "各項目を1行に制限するか"
            }
        }

# === 章立て・セクション・見出し関連の指示 ===

class StructuredSectionsChecker(MeetingInstruction):
    """指定された章立て構造をチェック"""
    
    def generate_instruction(self, sections: List[str] = None, use_markdown: bool = True, format_type: str = "numbered", **kwargs) -> str:
        if sections is None:
            sections = ["概要", "議論", "決定事項"]
        
        sections_str = " / ".join([f"{i+1}. {section}" for i, section in enumerate(sections)])
        base = f"{sections_str} の順で構成してください。"
        
        if use_markdown:
            base += " 各セクションは見出し（##）を付けてください。"
        
        return base
    
    def check_following(self, value: str) -> bool:
        sections = self._cached_kwargs.get('sections', ["概要", "議論", "決定事項"]) if self._cached_kwargs else ["概要", "議論", "決定事項"]
        use_markdown = self._cached_kwargs.get('use_markdown', True) if self._cached_kwargs else True
        
        # 各セクションが順番通りに含まれているかチェック
        last_index = -1
        for i, section in enumerate(sections):
            if use_markdown:
                # Markdownの見出し形式
                patterns = [
                    f"## {section}",
                    f"## {i+1}. {section}",
                    f"##{section}",
                    f"##{i+1}. {section}"
                ]
            else:
                # 通常の番号付き形式
                patterns = [
                    f"{i+1}. {section}",
                    f"{i+1}．{section}",
                    f"【{section}】"
                ]
            
            found = False
            for pattern in patterns:
                index = value.find(pattern)
                if index > last_index:
                    last_index = index
                    found = True
                    break
            
            if not found:
                return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "sections": {
                "type": "list",
                "default": ["概要", "議論", "決定事項"],
                "description": "セクションのリスト"
            },
            "use_markdown": {
                "type": "bool",
                "default": True,
                "description": "Markdown形式を使用するか"
            },
            "format_type": {
                "type": "str",
                "default": "numbered",
                "choices": ["numbered", "bulleted", "titled"],
                "description": "セクションの形式"
            }
        }

class QAFormatChecker(MeetingInstruction):
    """Q&A形式をチェック"""
    
    def generate_instruction(self, min_qa_pairs: int = 3, **kwargs) -> str:
        return f"Q&A形式で{min_qa_pairs}組以上の質問と回答をまとめてください。"
    
    def check_following(self, value: str) -> bool:
        # Q&Aのパターン
        q_patterns = [
            r'^\s*Q\d*[:：]',
            r'^\s*質問\d*[:：]',
            r'^\s*[Q【].*?[】][:：]?'
        ]
        
        a_patterns = [
            r'^\s*A\d*[:：]',
            r'^\s*回答\d*[:：]',
            r'^\s*[A【].*?[】][:：]?'
        ]
        
        lines = value.split('\n')
        q_count = 0
        a_count = 0
        
        for line in lines:
            for pattern in q_patterns:
                if re.match(pattern, line):
                    q_count += 1
                    break
            
            for pattern in a_patterns:
                if re.match(pattern, line):
                    a_count += 1
                    break
        
        min_pairs = self._cached_kwargs.get('min_qa_pairs', 3) if self._cached_kwargs else 3
        
        # Q&Aのペア数が最小数以上で、QとAの数が同じかチェック
        return q_count >= min_pairs and q_count == a_count
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "min_qa_pairs": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "最小Q&Aペア数"
            }
        }

class RequiredItemsChecker(MeetingInstruction):
    """必須項目の包含をチェック"""
    
    def generate_instruction(self, required_items: List[str] = None, **kwargs) -> str:
        if required_items is None:
            required_items = ["出席者", "議題", "議事内容", "決定事項", "次回アクション"]
        
        items_str = "】【".join(required_items)
        return f"以下の項目を含めてください：【{items_str}】"
    
    def check_following(self, value: str) -> bool:
        items = self._cached_kwargs.get('required_items', ["出席者", "議題", "議事内容", "決定事項", "次回アクション"]) if self._cached_kwargs else ["出席者", "議題", "議事内容", "決定事項", "次回アクション"]
        
        for item in items:
            # 項目が何らかの形式で含まれているかチェック
            patterns = [
                f"【{item}】",
                f"■{item}",
                f"# {item}",
                f"## {item}",
                f"{item}:",
                f"{item}："
            ]
            
            found = False
            for pattern in patterns:
                if pattern in value:
                    found = True
                    break
            
            if not found:
                return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "required_items": {
                "type": "list",
                "default": ["出席者", "議題", "議事内容", "決定事項", "次回アクション"],
                "description": "必須項目のリスト"
            }
        }


# === 内容評価関連の指示 ===

class EvaluationGradeChecker(MeetingInstruction):
    """要約後の評価（A/B/C）記載をチェック"""
    
    def generate_instruction(self, grade_format: str = "ABC", position: str = "end", **kwargs) -> str:
        if grade_format == "ABC":
            return "要約後に評価（A/B/C）を1つだけ記載してください。"
        elif grade_format == "number":
            return "要約後に評価（1/2/3/4/5）を1つだけ記載してください。"
        else:
            return "要約後に総合評価を記載してください。"
    
    def check_following(self, value: str) -> bool:
        grade_format = self._cached_kwargs.get('grade_format', 'ABC') if self._cached_kwargs else 'ABC'
        
        if grade_format == "ABC":
            # A/B/Cの評価が含まれているかチェック
            pattern = r'評価[:：]?\s*[A-C](?![A-Z0-9])|^[A-C]$|\n[A-C]$'
            matches = re.findall(pattern, value, re.MULTILINE)
            
            # 1つだけ記載されているか確認
            if len(matches) == 1:
                return True
            
            # 括弧内の評価もチェック
            bracket_pattern = r'[（\(][A-C][）\)]'
            bracket_matches = re.findall(bracket_pattern, value)
            return len(matches) + len(bracket_matches) == 1
            
        elif grade_format == "number":
            # 1-5の数字評価をチェック
            pattern = r'評価[:：]?\s*[1-5](?![0-9])|^[1-5]$|\n[1-5]$'
            matches = re.findall(pattern, value, re.MULTILINE)
            return len(matches) == 1
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "grade_format": {
                "type": "str",
                "default": "ABC",
                "choices": ["ABC", "number", "custom"],
                "description": "評価形式"
            },
            "position": {
                "type": "str",
                "default": "end",
                "choices": ["end", "beginning"],
                "description": "評価記載位置"
            }
        }

class FivePointScaleChecker(MeetingInstruction):
    """5段階評価スコアの冒頭記載をチェック"""
    
    def generate_instruction(self, scale_format: str = "stars", **kwargs) -> str:
        if scale_format == "stars":
            return "5段階評価のスコアを冒頭に★マークで記載してください。"
        elif scale_format == "points":
            return "5段階評価のスコア（1-5点）を冒頭に記載してください。"
        else:
            return "5段階評価のスコアを冒頭に記載してください。"
    
    def check_following(self, value: str) -> bool:
        scale_format = self._cached_kwargs.get('scale_format', 'stars') if self._cached_kwargs else 'stars'
        
        # 最初の100文字以内で評価を探す（冒頭判定）
        first_part = value[:100]
        
        if scale_format == "stars":
            # ★マークのパターン
            star_patterns = [
                r'^[\s　]*[★☆]{1,5}',  # 冒頭の★マーク
                r'^[\s　]*評価[:：]?\s*[★☆]{1,5}',  # 「評価：★★★」形式
                r'^[\s　]*\d+段階評価[:：]?\s*[★☆]{1,5}'  # 「5段階評価：★★★」形式
            ]
            
            for pattern in star_patterns:
                if re.search(pattern, first_part):
                    return True
                    
        elif scale_format == "points":
            # 数字評価のパターン
            point_patterns = [
                r'^[\s　]*[1-5]点',
                r'^[\s　]*[1-5]/5',
                r'^[\s　]*評価[:：]?\s*[1-5]点',
                r'^[\s　]*5段階評価[:：]?\s*[1-5]'
            ]
            
            for pattern in point_patterns:
                if re.search(pattern, first_part):
                    return True
        
        return False
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "scale_format": {
                "type": "str",
                "default": "stars",
                "choices": ["stars", "points", "mixed"],
                "description": "5段階評価の表記形式"
            }
        }

# === JSON出力関連の指示 ===

class JsonFormatChecker(MeetingInstruction):
    """JSON形式での出力をチェック"""
    
    def generate_instruction(self, json_schema: str = "summary_items", **kwargs) -> str:
        if json_schema == "summary_items":
            return 'JSON形式で出力してください：{"summary":"...","items":[{"topic":"...","points":["..."]}]}'
        elif json_schema == "summary_sources":
            return 'JSON形式で出力してください：{"summary":"...","sources":[{"id":1,"speaker":"A","message":"..."}]}'
        else:
            return "JSON形式で出力してください。"
    
    def check_following(self, value: str) -> bool:
        # JSONとして解析可能かチェック
        try:
            # コードブロックマーカーを除去
            cleaned_value = value.strip()
            if cleaned_value.startswith('```'):
                # ```json または ``` で始まる場合の処理
                lines = cleaned_value.split('\n')
                if len(lines) >= 3:  # 最低3行必要（開始、内容、終了）
                    cleaned_value = '\n'.join(lines[1:-1])
            
            parsed = json.loads(cleaned_value)
            
            json_schema = self._cached_kwargs.get('json_schema', 'summary_items') if self._cached_kwargs else 'summary_items'
            
            if json_schema == "summary_items":
                # 必須フィールドの確認
                if not isinstance(parsed, dict):
                    return False
                if 'summary' not in parsed or 'items' not in parsed:
                    return False
                if not isinstance(parsed['items'], list):
                    return False
                
                # items配列の各要素をチェック
                for item in parsed['items']:
                    if not isinstance(item, dict):
                        return False
                    if 'topic' not in item or 'points' not in item:
                        return False
                    if not isinstance(item['points'], list):
                        return False
                
                return True
                
            elif json_schema == "summary_sources":
                # 必須フィールドの確認
                if not isinstance(parsed, dict):
                    return False
                if 'summary' not in parsed or 'sources' not in parsed:
                    return False
                if not isinstance(parsed['sources'], list):
                    return False
                
                # sources配列の各要素をチェック
                for source in parsed['sources']:
                    if not isinstance(source, dict):
                        return False
                    if 'id' not in source or 'speaker' not in source or 'message' not in source:
                        return False
                    if not isinstance(source['id'], (int, float)):
                        return False
                
                return True
            
            else:
                # 一般的なJSON形式であればOK
                return True
                
        except (json.JSONDecodeError, ValueError):
            return False
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "json_schema": {
                "type": "str",
                "default": "summary_items",
                "choices": ["summary_items", "summary_sources", "custom"],
                "description": "JSONスキーマの種類"
            }
        }

# === 文体・敬語関連の指示 ===

class WritingStyleChecker(MeetingInstruction):
    """文体の統一をチェック"""
    
    def generate_instruction(self, style_type: str = "desu_masu", **kwargs) -> str:
        if style_type == "desu_masu":
            return "です・ます調で統一してください。"
        elif style_type == "de_aru":
            return "である調で統一してください。"
        elif style_type == "jotai":
            return "常体で統一してください。"
        elif style_type == "keitai":
            return "敬体にすることを徹底してください。"
        else:
            return "文体を統一してください。"
    
    def check_following(self, value: str) -> bool:
        style_type = self._cached_kwargs.get('style_type', 'desu_masu') if self._cached_kwargs else 'desu_masu'
        
        # 文末表現を抽出
        sentences = re.split(r'[。！？\n]', value)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return True
        
        if style_type == "desu_masu":
            # です・ます調のチェック
            desu_masu_endings = ['です', 'ます', 'ません', 'でした', 'ました', 'でしょう', 'ましょう']
            non_desu_masu_endings = ['である', 'だ', 'る', 'た', 'ない', 'だろう']
            
            for sentence in sentences:
                # 文末が適切な形式で終わっているかチェック
                has_desu_masu = any(sentence.endswith(ending) for ending in desu_masu_endings)
                has_non_desu_masu = any(sentence.endswith(ending) for ending in non_desu_masu_endings)
                
                # 混在している場合はFalse
                if has_non_desu_masu and not has_desu_masu:
                    return False
                    
        elif style_type == "de_aru":
            # である調のチェック
            de_aru_endings = ['である', 'だ', 'のである', 'のだ']
            non_de_aru_endings = ['です', 'ます', 'ません', 'でした', 'ました']
            
            for sentence in sentences:
                has_de_aru = any(sentence.endswith(ending) for ending in de_aru_endings)
                has_non_de_aru = any(sentence.endswith(ending) for ending in non_de_aru_endings)
                
                if has_non_de_aru:
                    return False
                    
        elif style_type == "jotai":
            # 常体のチェック（動詞の基本形、形容詞の基本形）
            jotai_endings = ['る', 'う', 'く', 'ぐ', 'す', 'つ', 'ぬ', 'ぶ', 'む', 'い', 'た', 'だ']
            keitai_endings = ['です', 'ます', 'ございます']
            
            for sentence in sentences:
                if any(sentence.endswith(ending) for ending in keitai_endings):
                    return False
                    
        elif style_type == "keitai":
            # 敬体のチェック
            keitai_indicators = ['です', 'ます', 'ございます', 'いたします', 'おります', 'いただく', 'くださる']
            
            # 少なくともいくつかの文で敬体が使われているかチェック
            keitai_count = 0
            for sentence in sentences:
                if any(indicator in sentence for indicator in keitai_indicators):
                    keitai_count += 1
            
            # 文の半分以上で敬体が使われていればOK
            return keitai_count >= len(sentences) * 0.5
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "style_type": {
                "type": "str",
                "default": "desu_masu",
                "choices": ["desu_masu", "de_aru", "jotai", "keitai"],
                "description": "文体の種類"
            }
        }

class NumberFormatChecker(MeetingInstruction):
    """数値の全角/半角統一をチェック"""
    
    def generate_instruction(self, format_type: str = "zenkaku", **kwargs) -> str:
        if format_type == "zenkaku":
            return "数値は全角で統一すること。"
        elif format_type == "hankaku":
            return "数値は半角にすること。"
        else:
            return "数値の表記を統一すること。"
    
    def check_following(self, value: str) -> bool:
        format_type = self._cached_kwargs.get('format_type', 'zenkaku') if self._cached_kwargs else 'zenkaku'
        
        # 全角数字と半角数字のパターン
        zenkaku_digits = '０１２３４５６７８９'
        hankaku_digits = '0123456789'
        
        has_zenkaku = any(char in value for char in zenkaku_digits)
        has_hankaku = any(char in value for char in hankaku_digits)
        
        if format_type == "zenkaku":
            # 半角数字が含まれていないことを確認
            return not has_hankaku or (has_zenkaku and not has_hankaku)
        elif format_type == "hankaku":
            # 全角数字が含まれていないことを確認
            return not has_zenkaku or (has_hankaku and not has_zenkaku)
        
        # どちらかに統一されていればOK
        return (has_zenkaku and not has_hankaku) or (has_hankaku and not has_zenkaku)
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "format_type": {
                "type": "str",
                "default": "zenkaku",
                "choices": ["zenkaku", "hankaku"],
                "description": "数値の形式"
            }
        }

# === 表形式関連の指示 ===

class MarkdownTableChecker(MeetingInstruction):
    """Markdown表形式での出力をチェック"""
    
    def generate_instruction(self, columns: List[str] = None, **kwargs) -> str:
        if columns is None:
            columns = ["論点", "結論", "担当", "期限"]
        
        columns_str = "｜".join(columns)
        return f"Markdown表（{columns_str}）で出力してください。"
    
    def check_following(self, value: str) -> bool:
        # Markdownテーブルの基本構造をチェック
        lines = value.strip().split('\n')
        
        # 最低3行必要（ヘッダー、区切り線、データ行）
        if len(lines) < 3:
            return False
        
        # パイプ（|）で区切られているかチェック
        table_lines = []
        for line in lines:
            if '|' in line or '｜' in line:
                table_lines.append(line)
        
        if len(table_lines) < 3:
            return False
        
        # ヘッダー行のチェック
        header_line = table_lines[0]
        
        # 指定されたカラムが含まれているかチェック
        columns = self._cached_kwargs.get('columns', ["論点", "結論", "担当", "期限"]) if self._cached_kwargs else ["論点", "結論", "担当", "期限"]
        
        for column in columns:
            if column not in header_line:
                return False
        
        # 区切り線のチェック（2行目）
        separator_line = table_lines[1]
        if not re.search(r'[-:\s|｜]+', separator_line):
            return False
        
        # データ行が存在するかチェック
        if len(table_lines) < 3:
            return False
        
        # 各行のカラム数が一致しているかチェック
        header_cols = len(re.split(r'[|｜]', header_line.strip()))
        for line in table_lines[2:]:
            line_cols = len(re.split(r'[|｜]', line.strip()))
            # カラム数の差が1以内なら許容（末尾の|の有無による）
            if abs(line_cols - header_cols) > 1:
                return False
        
        return True
    
    def get_instruction_args(self) -> Dict[str, Any]:
        return {
            "columns": {
                "type": "list",
                "default": ["論点", "結論", "担当", "期限"],
                "description": "テーブルのカラム名リスト"
            }
        }


#----J-MeetEval----------


