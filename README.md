# AITu2 - AI VTuber System

完全無料のAI VTuberシステム。単一画像からリアルタイムでキャラクターをアニメーション化し、コメントに応答して配信します。

## 機能

- **画像認識AI**: Gemini 2.5 Flashによるキャラクター状態分析
- **会話AI**: OpenAI GPT-4o-miniによる自然な応答
- **音声生成**: FishAudioによる可愛い女の子の声
- **アニメーション**: 呼吸・口の動き・感情表現
- **配信**: OBS Studio + RTMP対応

## 必要な環境

- Python 3.8+
- NVIDIA GPU (ComfyUI用)
- FFmpeg

## セットアップ

1. 依存関係インストール
```bash
pip install -r requirements.txt
```

2. 環境変数設定
```bash
# .envファイルを作成
OPENAI_API_KEY=your_openai_key
FISHAUDIO_API_KEY=your_fishaudio_key
GEMINI_API_KEY=your_gemini_key
```

3. ComfyUI起動
```bash
cd ComfyUI
python main.py --listen 127.0.0.1 --port 8188
```

4. システム起動
```bash
python phase4_complete_integration.py
```

## ファイル構成

- `phase4_complete_integration.py`: メインシステム
- `yume_image.png`: キャラクター画像
- `# 要件定義書 - AI VTuber開発プロジェクト.txt`: 要件定義書

## ライセンス

MIT License
