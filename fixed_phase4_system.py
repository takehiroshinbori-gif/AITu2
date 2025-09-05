#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版Phase 4: 完全統合AI VTuberシステム
- FishAudio音声の正しい再生
- 感情タグの除去
- 一時ファイルの適切な管理
- GUI表示の修正
"""

import os
import sys
import time
import math
import logging
import threading
import tempfile
import subprocess
from typing import Optional, Dict, Any
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import requests
from dotenv import load_dotenv
from openai import OpenAI
import pygame

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

class FixedCompleteAIVTuberSystem:
    """修正版完全統合AI VTuberシステム"""
    def __init__(self):
        # 音声システム
        self.api_key = os.getenv("FISHAUDIO_API_KEY")
        self.voice_id = os.getenv("FISHAUDIO_VOICE_ID", "e32d8978e5b740058b87310599f15b4d")
        self.endpoint = "https://api.fish.audio/v1/tts"
        self.session = requests.Session()
        self.is_playing = False
        
        # pygame音声システム初期化
        pygame.mixer.init()
        
        # AI会話システム
        self.openai_client = OpenAI()
        
        # アニメーションシステム
        self.image_path = "yume_image.png"
        self.original_image = None
        self.animation_state = {
            "breathing": 0.0,
            "talking": False,
            "emotion": "neutral",
            "idle_timer": 0.0,
            "blink_timer": 0.0
        }
        
        # アニメーションパラメータ
        self.breathing_speed = 0.02
        self.breathing_amplitude = 0.05
        self.talking_amplitude = 0.1
        self.blink_interval = 3.0
        
        # 記憶システム
        self.memory = {
            "profile": "",
            "conversation_history": [],
            "emotion_history": [],
            "last_emotion": "neutral"
        }
        
        # GUI
        self.root = None
        self.character_label = None
        self.comment_entry = None
        self.response_label = None
        self.emotion_label = None
        self.streaming_label = None
        
        # 状態管理
        self.is_streaming = False
        self.is_processing = False
        self.animation_running = False
        
        self.load_image()
        logger.info("修正版完全統合AI VTuberシステム初期化完了")
    
    def load_image(self):
        """画像読み込み"""
        try:
            if os.path.exists(self.image_path):
                self.original_image = Image.open(self.image_path)
                logger.info(f"✅ 画像読み込み成功: {self.image_path}")
            else:
                logger.error(f"❌ 画像ファイルが見つかりません: {self.image_path}")
        except Exception as e:
            logger.error(f"❌ 画像読み込みエラー: {e}")
    
    def get_sentiment(self, text: str) -> str:
        """感情検出"""
        positive_words = ["楽しい", "嬉しい", "好き", "最高", "すごい", "面白い", "可愛い", "笑", "ナイス", "ありがと", "助かる"]
        negative_words = ["嫌い", "悲しい", "辛い", "最悪", "怖い", "怒り", "怒", "むかつく", "ごめん", "謝"]
        surprise_words = ["驚", "びっくり", "!?", "！?"]
        
        text_lower = text.lower()
        
        if any(word in text for word in positive_words):
            return "happy"
        elif any(word in text for word in negative_words):
            return "sad"
        elif any(word in text for word in surprise_words):
            return "surprised"
        else:
            return "neutral"
    
    def clean_response_text(self, text: str) -> str:
        """応答テキストから感情タグを除去"""
        # 絵文字を除去（ただし、テキストが空にならないように注意）
        import re
        # 基本的な絵文字パターンを除去
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        cleaned_text = emoji_pattern.sub('', text)
        cleaned_text = cleaned_text.strip()
        
        # テキストが空になった場合は元のテキストを返す
        if not cleaned_text:
            return text.strip()
        
        return cleaned_text
    
    def generate_response(self, user_input: str) -> str:
        """AI応答生成"""
        try:
            # システムプロンプト（絵文字なし）
            system_prompt = (
                "あなたは可愛い女の子のAIアシスタント『もも』。"
                "親しみやすいタメ口で、絵文字は使わない。"
                "会話の状況に合わせて言葉づかいで感情表現する。"
                "1〜2文・最大80文字・テンポ良く短く返す。"
            )
            
            # 会話履歴を構築
            messages = [{"role": "system", "content": system_prompt}]
            
            # 記憶を追加
            if self.memory["profile"]:
                messages.append({"role": "system", "content": f"ユーザーのプロフィール: {self.memory['profile']}"})
            
            # 会話履歴を追加（最新10件）
            for msg in self.memory["conversation_history"][-10:]:
                messages.append(msg)
            
            # 現在のユーザー入力
            messages.append({"role": "user", "content": user_input})
            
            # OpenAI API呼び出し
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.8,
                max_tokens=160
            )
            
            reply = response.choices[0].message.content.strip()
            
            # 感情タグを除去
            cleaned_reply = self.clean_response_text(reply)
            
            # 会話履歴に追加
            self.memory["conversation_history"].append({"role": "user", "content": user_input})
            self.memory["conversation_history"].append({"role": "assistant", "content": cleaned_reply})
            
            # 履歴を最新10件に制限
            self.memory["conversation_history"] = self.memory["conversation_history"][-10:]
            
            logger.info(f"✅ AI応答生成: {cleaned_reply}")
            return cleaned_reply
            
        except Exception as e:
            logger.error(f"❌ AI応答生成エラー: {e}")
            return "うん、ちょっと考えてみるね..."
    
    def generate_audio(self, text: str, emotion: str = "neutral") -> Optional[bytes]:
        """音声生成"""
        if not self.api_key:
            logger.error("FishAudio API Keyが設定されていません")
            return None
        
        # 感情に応じたパラメータ調整
        emotion_params = {
            "happy": {"speed": 1.1, "pitch": 1.05},
            "sad": {"speed": 0.9, "pitch": 0.95},
            "angry": {"speed": 1.05, "pitch": 1.02},
            "surprised": {"speed": 1.15, "pitch": 1.08},
            "neutral": {"speed": 1.0, "pitch": 1.0}
        }
        
        params = emotion_params.get(emotion, emotion_params["neutral"])
        
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "format": "mp3",
            "speed": params["speed"],
            "pitch": params["pitch"]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                logger.info(f"✅ FishAudio音声生成成功: {text[:20]}...")
                return response.content
            else:
                logger.error(f"❌ FishAudio音声生成失敗: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ FishAudio音声生成エラー: {e}")
            return None
    
    def play_audio_with_pygame(self, audio_data: bytes) -> bool:
        """pygameを使用した音声再生"""
        try:
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            # 非同期で音声再生
            def play_audio():
                try:
                    self.is_playing = True
                    
                    # pygameで音声再生
                    pygame.mixer.music.load(tmp_file_path)
                    pygame.mixer.music.play()
                    
                    # 再生完了を待つ
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # 一時ファイル削除
                    try:
                        os.unlink(tmp_file_path)
                        logger.info("✅ 一時ファイル削除完了")
                    except Exception as e:
                        logger.error(f"❌ 一時ファイル削除エラー: {e}")
                    
                    self.is_playing = False
                    logger.info("✅ FishAudio音声再生完了")
                    
                except Exception as e:
                    logger.error(f"❌ pygame音声再生エラー: {e}")
                    self.is_playing = False
            
            # 別スレッドで再生
            threading.Thread(target=play_audio, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"❌ pygame音声再生準備エラー: {e}")
            return False
    
    def apply_breathing_animation(self, image: Image.Image, phase: float) -> Image.Image:
        """呼吸アニメーション適用"""
        try:
            # 呼吸による微細な拡大縮小
            scale = 1.0 + math.sin(phase) * self.breathing_amplitude
            
            # 画像サイズ計算
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # サイズが0以下にならないようにチェック
            if new_width <= 0 or new_height <= 0:
                return image
            
            # リサイズ
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 中央に配置
            result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2
            result.paste(resized, (x_offset, y_offset))
            
            return result
            
        except Exception as e:
            logger.error(f"呼吸アニメーションエラー: {e}")
            return image
    
    def apply_talking_animation(self, image: Image.Image, intensity: float) -> Image.Image:
        """話す時のアニメーション適用"""
        try:
            # 話す時の微細な動き
            move_x = math.sin(time.time() * 10) * intensity * 2
            move_y = math.sin(time.time() * 8) * intensity * 1
            
            # 画像を移動
            result = Image.new('RGBA', image.size, (0, 0, 0, 0))
            result.paste(image, (int(move_x), int(move_y)))
            
            return result
            
        except Exception as e:
            logger.error(f"話すアニメーションエラー: {e}")
            return image
    
    def apply_emotion_animation(self, image: Image.Image, emotion: str) -> Image.Image:
        """感情アニメーション適用"""
        try:
            result = image.copy()
            
            # 感情に応じた色調整
            if emotion == "happy":
                # 明るくする
                result = result.convert('HSV')
                h, s, v = result.split()
                v = v.point(lambda x: min(255, x * 1.1))
                result = Image.merge('HSV', (h, s, v)).convert('RGB')
            elif emotion == "sad":
                # 暗くする
                result = result.convert('HSV')
                h, s, v = result.split()
                v = v.point(lambda x: max(0, x * 0.9))
                result = Image.merge('HSV', (h, s, v)).convert('RGB')
            elif emotion == "angry":
                # 赤みを強くする
                result = result.convert('HSV')
                h, s, v = result.split()
                h = h.point(lambda x: (x + 10) % 360)
                s = s.point(lambda x: min(255, x * 1.2))
                result = Image.merge('HSV', (h, s, v)).convert('RGB')
            
            return result
            
        except Exception as e:
            logger.error(f"感情アニメーションエラー: {e}")
            return image
    
    def generate_animated_frame(self) -> Optional[Image.Image]:
        """アニメーションフレーム生成"""
        if not self.original_image:
            return None
        
        try:
            # 基本画像をコピー
            frame = self.original_image.copy()
            
            # 呼吸アニメーション
            breathing_phase = time.time() * self.breathing_speed
            frame = self.apply_breathing_animation(frame, breathing_phase)
            
            # 話す時のアニメーション
            if self.animation_state["talking"]:
                talking_intensity = 0.5 + 0.5 * math.sin(time.time() * 8)
                frame = self.apply_talking_animation(frame, talking_intensity)
            
            # 感情アニメーション
            frame = self.apply_emotion_animation(frame, self.animation_state["emotion"])
            
            return frame
            
        except Exception as e:
            logger.error(f"アニメーションフレーム生成エラー: {e}")
            return self.original_image
    
    def update_animation_state(self, talking: bool = False, emotion: str = "neutral"):
        """アニメーション状態更新"""
        self.animation_state["talking"] = talking
        self.animation_state["emotion"] = emotion
        self.animation_state["idle_timer"] += 0.016  # 60fps想定
        self.animation_state["blink_timer"] += 0.016
        
        # 瞬きタイマーリセット
        if self.animation_state["blink_timer"] >= self.blink_interval:
            self.animation_state["blink_timer"] = 0.0
    
    def process_comment(self, comment: str):
        """コメント処理"""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            logger.info(f"コメント処理開始: {comment}")
            
            # 感情検出
            emotion = self.get_sentiment(comment)
            self.memory["last_emotion"] = emotion
            
            # アニメーション状態更新
            self.update_animation_state(talking=True, emotion=emotion)
            
            # AI応答生成
            response = self.generate_response(comment)
            
            # GUI更新
            self.update_gui(response, emotion)
            
            # 音声生成と再生
            self.generate_and_play_audio(response, emotion)
            
            # アニメーション状態を待機に戻す
            self.update_animation_state(talking=False, emotion=emotion)
            
            logger.info(f"コメント処理完了: {comment}")
            
        except Exception as e:
            logger.error(f"❌ コメント処理エラー: {e}")
        finally:
            self.is_processing = False
    
    def generate_and_play_audio(self, text: str, emotion: str):
        """音声生成と再生"""
        try:
            # FishAudioで音声生成
            audio_data = self.generate_audio(text, emotion)
            
            if audio_data:
                # pygameで音声再生
                self.play_audio_with_pygame(audio_data)
            else:
                logger.error("❌ FishAudio音声生成失敗")
                
        except Exception as e:
            logger.error(f"❌ FishAudio音声生成・再生エラー: {e}")
    
    def update_gui(self, response: str, emotion: str):
        """GUI更新"""
        try:
            if self.response_label:
                self.response_label.config(text=f"もも: {response}")
            
            if self.emotion_label:
                self.emotion_label.config(text=emotion)
                
        except Exception as e:
            logger.error(f"❌ GUI更新エラー: {e}")
    
    def update_character_display(self):
        """キャラクター表示更新"""
        try:
            if self.character_label:
                # アニメーションフレーム生成
                frame = self.generate_animated_frame()
                
                if frame:
                    # PIL画像をTkinter用に変換
                    photo = ImageTk.PhotoImage(frame)
                    self.character_label.config(image=photo)
                    self.character_label.image = photo  # 参照を保持
                    
        except Exception as e:
            logger.error(f"❌ キャラクター表示更新エラー: {e}")
    
    def start_streaming(self):
        """ストリーミング開始"""
        self.is_streaming = True
        if self.streaming_label:
            self.streaming_label.config(text="ストリーミング中")
        logger.info("✅ ストリーミング開始")
    
    def stop_streaming(self):
        """ストリーミング停止"""
        self.is_streaming = False
        if self.streaming_label:
            self.streaming_label.config(text="ストリーミング停止")
        logger.info("✅ ストリーミング停止")
    
    def create_gui(self):
        """GUI作成"""
        self.root = tk.Tk()
        self.root.title("修正版完全統合AI VTuberシステム - もも")
        self.root.geometry("1200x800")
        
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側：キャラクター表示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # キャラクター画像表示
        self.character_label = ttk.Label(left_frame, text="キャラクター読み込み中...")
        self.character_label.pack(fill=tk.BOTH, expand=True)
        
        # 右側：コントロールパネル
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # タイトル
        title_label = ttk.Label(right_frame, text="修正版完全統合AI VTuberシステム", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # コメント入力
        ttk.Label(right_frame, text="コメント:").pack(anchor=tk.W)
        self.comment_entry = ttk.Entry(right_frame, width=30)
        self.comment_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 送信ボタン
        send_button = ttk.Button(right_frame, text="送信", command=self.send_comment)
        send_button.pack(pady=(0, 20))
        
        # ももの返答
        ttk.Label(right_frame, text="ももの返答:").pack(anchor=tk.W)
        self.response_label = ttk.Label(right_frame, text="もも: こんにちは！", wraplength=250)
        self.response_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 現在の感情
        ttk.Label(right_frame, text="現在の感情:").pack(anchor=tk.W)
        self.emotion_label = ttk.Label(right_frame, text="neutral", font=("Arial", 12, "bold"))
        self.emotion_label.pack(anchor=tk.W, pady=(0, 20))
        
        # ストリーミング
        ttk.Label(right_frame, text="ストリーミング:").pack(anchor=tk.W)
        self.streaming_label = ttk.Label(right_frame, text="ストリーミング停止")
        self.streaming_label.pack(anchor=tk.W, pady=(0, 10))
        
        # ストリーミングボタン
        streaming_button = ttk.Button(right_frame, text="ストリーミング開始/停止", command=self.toggle_streaming)
        streaming_button.pack(pady=(0, 20))
        
        # 終了ボタン
        exit_button = ttk.Button(right_frame, text="終了", command=self.root.quit)
        exit_button.pack(side=tk.BOTTOM)
        
        # 初期化
        self.update_character_display()
        
        # アニメーション更新ループ
        self.animation_loop()
    
    def send_comment(self):
        """コメント送信"""
        comment = self.comment_entry.get().strip()
        if comment:
            logger.info(f"コメント送信: {comment}")
            self.comment_entry.delete(0, tk.END)
            
            # 別スレッドでコメント処理
            threading.Thread(target=self.process_comment, args=(comment,), daemon=True).start()
    
    def toggle_streaming(self):
        """ストリーミング切り替え"""
        if self.is_streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
    
    def animation_loop(self):
        """アニメーションループ"""
        try:
            if self.animation_running:
                self.update_character_display()
        except Exception as e:
            logger.error(f"❌ アニメーションループエラー: {e}")
        
        # 60fpsで更新
        if self.root and self.root.winfo_exists():
            self.root.after(16, self.animation_loop)
    
    def start_animation(self):
        """アニメーション開始"""
        self.animation_running = True
        logger.info("✅ アニメーション開始")
    
    def stop_animation(self):
        """アニメーション停止"""
        self.animation_running = False
        logger.info("✅ アニメーション停止")
    
    def run(self):
        """システム実行"""
        try:
            logger.info("修正版完全統合AI VTuberシステム開始")
            self.create_gui()
            self.start_animation()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"❌ システム実行エラー: {e}")
        finally:
            self.stop_animation()
            logger.info("修正版完全統合AI VTuberシステム終了")

def main():
    """メイン実行"""
    logger.info("修正版Phase 4: 完全統合AI VTuberシステム開始")
    
    system = FixedCompleteAIVTuberSystem()
    system.run()

if __name__ == "__main__":
    main()
