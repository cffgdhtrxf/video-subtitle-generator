# 视频字幕生成工作流

## 环境

- Python: `venv/Scripts/python.exe`（虚拟环境，不要用系统 Python）
- GPU: RTX 3070 Ti Laptop (8GB VRAM)，驱动版本 512.78 **太旧**，需 >= 522.06 才能用 CUDA
- 当前只能用 CPU 模式（`--device cpu --compute-type int8`）

## 字幕生成脚本

```
python subtitle_generator.py [视频文件] [--speakers] [--device cuda] [--compute-type float16] [--hf-token xxx]
```

- 默认输出到视频同目录，同名 `.srt`
- `--speakers` 启用说话人分离（需要 HF token）
- 驱动更新后可用 `--device cuda --compute-type float16` 大幅加速

## HuggingFace 网络

- huggingface.co 被墙，用镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- HF token 存在项目目录的 `.hf_token` 文件里
- pyannote 模型较大，首次下载不稳定，多试几次或换镜像

## 代码修复记录

- `diarize_speakers` 函数的 import bug 已修（`from whisperx.diarize import DiarizationPipeline` → `import whisperx` + `whisperx.diarize.DiarizationPipeline`）
- 模型从 `speaker-diarization-community-1` 改为 `speaker-diarization-3.1`（识别更准）

## 模型缓存

- Whisper large-v3: 首次下载约 3GB
- pyannote speaker-diarization-3.1: 需先接受 HF 协议才能用 token 下
- 对齐模型 wav2vec2: 按语言自动下载
