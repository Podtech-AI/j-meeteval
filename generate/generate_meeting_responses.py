#!/usr/bin/env python3
# coding=utf-8

"""
jMeet-Eval: 会議特化AIモデルの応答を生成してinput_response_data.jsonlを作成するスクリプト

使用例:
python generate_meeting_responses.py \
  --input_data=data/jmeet_input_data.jsonl \
  --output=data/jmeet_response_data_claude.jsonl \
  --model=claude-3-5-sonnet

サポートするモデル:
- OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-mini
- Anthropic: claude-3-5-sonnet, claude-3-haiku, claude-3-opus
- HuggingFace: sbintuitions/sarashina2.2-1b-instruct-v0.1, または任意のHFモデル
- ローカル: mock (会議特化テスト用のモック応答)
"""

import json
import argparse
import time
import os
import sys
from typing import List, Dict, Any
import logging
import re

# オプショナルなAPI依存関係
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeetingResponseGenerator:
    """会議特化AIモデルの応答を生成するクラス"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.openai_client = None
        self.anthropic_client = None
        self.hf_tokenizer = None
        self.hf_model = None
        self.hf_pipeline = None
        
        # モデルタイプを判定してクライアントを初期化
        if self._is_openai_model(model_name):
            self._init_openai()
        elif self._is_anthropic_model(model_name):
            self._init_anthropic()
        elif self._is_huggingface_model(model_name):
            self._init_huggingface()
        elif model_name == "mock":
            logger.info("会議特化モックモードで実行します")
        else:
            raise ValueError(f"サポートされていないモデル: {model_name}")
    
    def _is_openai_model(self, model_name: str) -> bool:
        """OpenAIモデルかどうかを判定"""
        openai_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        return any(model in model_name for model in openai_models)
    
    def _is_anthropic_model(self, model_name: str) -> bool:
        """Anthropicモデルかどうかを判定"""
        anthropic_models = ["claude-3", "claude-2", "claude-instant"]
        return any(model in model_name for model in anthropic_models)
    
    def _is_huggingface_model(self, model_name: str) -> bool:
        """HuggingFaceモデルかどうかを判定"""
        return (("/" in model_name and not model_name.startswith("http")) or 
                model_name.startswith("hf:") or
                model_name in ["sarashina", "sarashina2.2-1b-instruct"])
    
    def _init_openai(self):
        """OpenAIクライアントを初期化"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI パッケージがインストールされていません: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境変数が設定されていません")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAI クライアントを初期化しました: {self.model_name}")
    
    def _init_anthropic(self):
        """Anthropicクライアントを初期化"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic パッケージがインストールされていません: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 環境変数が設定されていません")
        
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Anthropic クライアントを初期化しました: {self.model_name}")
    
    def _init_huggingface(self):
        """HuggingFaceモデルを初期化"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers パッケージがインストールされていません: pip install torch transformers")
        
        # モデル名の正規化
        if self.model_name.startswith("hf:"):
            model_id = self.model_name[3:]
        elif self.model_name == "sarashina":
            model_id = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
        elif self.model_name == "sarashina2.2-1b-instruct":
            model_id = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
        else:
            model_id = self.model_name
        
        logger.info(f"HuggingFace モデルを読み込み中: {model_id}")
        
        # デバイス設定
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA が利用可能です。GPU を使用します。")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon Mac
                logger.info("MPS が利用可能です。Apple Silicon GPU を使用します。")
            else:
                self.device = "cpu"
                logger.info("GPU が利用できません。CPU を使用します。")
        
        try:
            # トークナイザーとモデルを読み込み
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # パディングトークンが設定されていない場合は設定
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            # モデル設定
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            }
            
            if self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
            
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.hf_model.to(self.device)
            
            # パイプラインを作成
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.hf_model,
                tokenizer=self.hf_tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            )
            
            logger.info(f"HuggingFace モデルを初期化しました: {model_id} (device: {self.device})")
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace モデルの初期化に失敗しました: {e}")
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """プロンプトに対する応答を生成"""
        for attempt in range(max_retries):
            try:
                if self._is_openai_model(self.model_name):
                    return self._generate_openai_response(prompt)
                elif self._is_anthropic_model(self.model_name):
                    return self._generate_anthropic_response(prompt)
                elif self._is_huggingface_model(self.model_name):
                    return self._generate_huggingface_response(prompt)
                elif self.model_name == "mock":
                    return self._generate_meeting_mock_response(prompt)
                else:
                    raise ValueError(f"サポートされていないモデル: {self.model_name}")
            
            except Exception as e:
                logger.warning(f"応答生成に失敗 (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数バックオフ
    
    def _generate_openai_response(self, prompt: str) -> str:
        """OpenAI APIで応答を生成"""
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "あなたは会議関連タスクの専門家です。与えられた指示に従って正確に回答し、指定された制約を必ず守ってください。会議の形式、構造、内容について適切な知識を活用してください。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    
    def _generate_anthropic_response(self, prompt: str) -> str:
        """Anthropic APIで応答を生成"""
        response = self.anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=4000,
            temperature=0.7,
            system="あなたは会議関連タスクの専門家です。与えられた指示に従って正確に回答し、指定された制約を必ず守ってください。会議の形式、構造、内容について適切な知識を活用してください。",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    
    def _generate_huggingface_response(self, prompt: str) -> str:
        """HuggingFace モデルで応答を生成"""
        # 会議特化プロンプトフォーマット
        if "sarashina" in self.model_name.lower() or "instruct" in self.model_name.lower():
            formatted_prompt = f"### 指示\n会議関連タスクの専門家として、以下の指示に従って回答してください。\n\n{prompt}\n\n### 回答\n"
        else:
            formatted_prompt = f"会議関連タスクの専門家として: {prompt}"
        
        try:
            # 生成パラメータ
            generation_kwargs = {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.hf_tokenizer.eos_token_id,
                "eos_token_id": self.hf_tokenizer.eos_token_id,
                "return_full_text": False,
            }
            
            # 応答生成
            outputs = self.hf_pipeline(formatted_prompt, **generation_kwargs)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                response = generated_text.strip()
                
                # 指示形式の場合、### 以降の部分を除去
                if "###" in response:
                    response = response.split("###")[0].strip()
                
                if not response:
                    response = "会議関連の適切な応答を生成できませんでした。"
                
                return response
            else:
                return "会議応答生成に失敗しました。"
                
        except Exception as e:
            logger.error(f"HuggingFace 応答生成エラー: {e}")
            raise
    
    def _generate_meeting_mock_response(self, prompt: str) -> str:
        """会議特化テスト用のモック応答を生成"""
        prompt_lower = prompt.lower()
        
        # 会議特化のモック応答テンプレート
        meeting_mock_responses = {
            "agenda": """1. 開会挨拶とメンバー紹介
2. 前回議事録の確認
3. 今期の業績報告
4. 来期の事業計画
5. 質疑応答とディスカッション
6. 次回会議の日程調整""",
            
            "participants": """参加者リスト:
• 田中部長 (営業部)
• 佐藤課長 (マーケティング部)
• 山田主任 (開発部)
• 鈴木係長 (企画部)""",
            
            "minutes": """【会議議事録】
日時: 2024年9月27日 14:00-15:30
場所: 第1会議室
参加者: 田中部長、佐藤課長、山田主任、鈴木係長

【主な議題】
1. 今期売上目標の進捗確認
2. 新商品企画の検討
3. 来期予算計画の策定

【決定事項】
- 来期予算は前年比110%で設定
- 新商品企画は12月末までに詳細決定""",
            
            "action_items": """アクションアイテム:
• 市場調査レポート作成 (担当: 佐藤課長, 期限: 10月15日)
• 予算案の詳細検討 (担当: 田中部長, 期限: 10月20日) 
• 新商品コンセプト資料準備 (担当: 山田主任, 期限: 10月25日)""",
            
            "decisions": """決定事項:
1. 来期売上目標を前年比115%に設定
2. 新商品開発プロジェクトを正式スタート
3. 次回会議を10月30日に開催""",
            
            "time": """会議時間:
開始時刻: 14:00
終了時刻: 15:30
日程: 2024年9月27日(金)""",
            
            "location": """会議場所:
場所: 本社第1会議室
形式: ハイブリッド開催(対面+オンライン)
オンライン: Zoom会議室""",
            
            "summary": """会議サマリー:
今期の業績は順調に推移しており目標達成見込み。来期に向けて新商品開発と予算計画を重点的に検討。各担当者のアクションアイテムを明確化し10月末までに進捗確認予定。""",
            
            "preparation": """会議準備項目:
□ 資料配布 (前日まで)
□ 会議室予約確認
□ オンライン環境テスト  
□ 参加者への事前通知
□ アジェンダ最終確認""",
            
            "next_steps": """次のステップ:
1. 今回のアクションアイテム実行
2. 10月30日に進捗確認会議
3. 来期計画の詳細検討継続
4. 関係部署への情報共有""",
            
            "feedback": """会議フィードバック:
評価 (5段階): 4
Plus: 活発な議論と具体的な決定
Delta: 時間管理の改善必要
次回改善点: より詳細な事前資料準備"""
        }
        
        # プロンプトの内容に基づいて適切なモック応答を選択
        if any(keyword in prompt_lower for keyword in ["アジェンダ", "agenda"]):
            return meeting_mock_responses["agenda"]
        elif any(keyword in prompt_lower for keyword in ["参加者", "participant"]):
            return meeting_mock_responses["participants"]
        elif any(keyword in prompt_lower for keyword in ["議事録", "minutes"]):
            return meeting_mock_responses["minutes"]
        elif any(keyword in prompt_lower for keyword in ["アクション", "action"]):
            return meeting_mock_responses["action_items"]
        elif any(keyword in prompt_lower for keyword in ["決定", "decision"]):
            return meeting_mock_responses["decisions"]
        elif any(keyword in prompt_lower for keyword in ["時間", "time", "日時"]):
            return meeting_mock_responses["time"]
        elif any(keyword in prompt_lower for keyword in ["場所", "location", "会議室"]):
            return meeting_mock_responses["location"]
        elif any(keyword in prompt_lower for keyword in ["サマリー", "要約", "summary"]):
            return meeting_mock_responses["summary"]
        elif any(keyword in prompt_lower for keyword in ["準備", "preparation"]):
            return meeting_mock_responses["preparation"]
        elif any(keyword in prompt_lower for keyword in ["次", "next", "ステップ"]):
            return meeting_mock_responses["next_steps"]
        elif any(keyword in prompt_lower for keyword in ["フィードバック", "feedback", "評価"]):
            return meeting_mock_responses["feedback"]
        else:
            # 汎用的な会議モック応答
            return f"[会議モック応答] {prompt[:100]}... に対する会議関連の応答です。これはjMeet-Eval用のテスト応答として生成されました。"


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """input_data.jsonlを読み込み"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"{len(data)}件の会議データを読み込みました: {input_file}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONファイルの形式が不正です: {e}")


def save_response_data(data: List[Dict[str, str]], output_file: str):
    """input_response_data.jsonl形式で保存"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"{len(data)}件の会議応答データを保存しました: {output_file}")
    except Exception as e:
        raise RuntimeError(f"ファイル保存に失敗しました: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="jMeet-Eval: 会議特化AIモデルの応答を生成してinput_response_data.jsonlを作成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Claudeで会議データの応答生成
  python generate_meeting_responses.py \\
    --input_data=data/jmeet_input_data.jsonl \\
    --output=data/jmeet_response_data_claude.jsonl \\
    --model=claude-3-5-sonnet

  # 会議特化モックモードでテスト
  python generate_meeting_responses.py \\
    --input_data=data/jmeet_input_data.jsonl \\
    --output=data/jmeet_response_data_mock.jsonl \\
    --model=mock

環境変数:
  OPENAI_API_KEY    - OpenAI APIキー (OpenAIモデル使用時)
  ANTHROPIC_API_KEY - Anthropic APIキー (Anthropicモデル使用時)
        """
    )
    
    parser.add_argument('--input_data', required=True, 
                       help='入力データファイル (jmeet_input_data.jsonl)')
    parser.add_argument('--output', required=True,
                       help='出力ファイル (input_response_data.jsonl)')
    parser.add_argument('--model', default='mock',
                       help='使用するモデル (default: mock)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='API呼び出し間の待機時間(秒) (default: 1.0)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='処理する最大項目数 (default: 全件)')
    parser.add_argument('--start_from', type=int, default=0,
                       help='開始項目番号 (default: 0)')
    parser.add_argument('--device', default='auto',
                       help='HuggingFace モデル用デバイス (auto/cpu/cuda/mps) (default: auto)')
    
    args = parser.parse_args()
    
    try:
        # 1. 入力データを読み込み
        logger.info("=== jMeet-Eval 応答生成開始 ===")
        input_data = load_input_data(args.input_data)
        
        # 処理範囲を制限
        if args.max_items:
            end_index = min(args.start_from + args.max_items, len(input_data))
            input_data = input_data[args.start_from:end_index]
            logger.info(f"処理範囲: {args.start_from} - {end_index-1} ({len(input_data)}件)")
        
        # 2. 会議応答ジェネレーターを初期化
        generator = MeetingResponseGenerator(args.model, device=args.device)
        
        # 3. 各プロンプトに対して応答生成
        response_data = []
        total_items = len(input_data)
        
        for i, item in enumerate(input_data):
            prompt = item['prompt']
            key = item.get('key', f'item_{i}')
            
            logger.info(f"処理中 ({i+1}/{total_items}) Key {key}: {prompt[:80]}...")
            
            try:
                # AIモデルに送信
                response = generator.generate_response(prompt)
                
                # input_response形式で保存
                response_data.append({
                    "prompt": prompt,
                    "response": response
                })
                
                logger.info(f"✅ 完了 Key {key}: {len(response)}文字の会議応答を生成")
                
                # レート制限対応
                if i < total_items - 1:  # 最後の項目でない場合
                    time.sleep(args.delay)
                    
            except Exception as e:
                logger.error(f"❌ 失敗 Key {key}: {e}")
                # エラーの場合はエラーメッセージを応答として記録
                response_data.append({
                    "prompt": prompt,
                    "response": f"[エラー] 会議応答生成に失敗しました: {str(e)}"
                })
        
        # 4. 結果を保存
        save_response_data(response_data, args.output)
        
        logger.info("=== jMeet-Eval 応答生成完了 ===")
        logger.info(f"成功: {len([r for r in response_data if not r['response'].startswith('[エラー]')])}件")
        logger.info(f"失敗: {len([r for r in response_data if r['response'].startswith('[エラー]')])}件")
        logger.info(f"出力ファイル: {args.output}")
        
    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()