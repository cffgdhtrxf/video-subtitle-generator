# Video Subtitle Generator 视频字幕生成器

基于 **WhisperX** + **pyannote.audio** 的自动字幕生成工具。支持中文和英文语音识别、词级对齐、说话人分离，输出标准 SRT 字幕文件。

## 功能

- 🎤 **自动语音识别** — 中/英文自动检测，基于 faster-whisper large-v3
- 🎯 **词级强制对齐** — wav2vec2 模型，精准到每个词的时间戳
- 👥 **说话人分离**（可选）— pyannote/speaker-diarization-3.1，区分不同说话人
- 🧹 **智能后处理** — 过滤填充词、去口吃、合并短句、孤词修复
- 📝 **分段优化** — 基于语义和时长的智能分组，字幕阅读体验更好

## 系统要求

- Python 3.8+
- ffmpeg（需单独安装）
  - **Windows**: `winget install ffmpeg` 或从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

## 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/video-subtitle-generator.git
cd video-subtitle-generator

# 安装依赖
pip install -r requirements.txt
```

## 用法

```bash
# 自动检测当前目录的视频文件
python subtitle_generator.py

# 指定视频文件
python subtitle_generator.py video.mp4

# 启用说话人分离（需要 HuggingFace Token）
python subtitle_generator.py video.mp4 --speakers --hf-token hf_xxxxxxx

# 使用 GPU 加速（NVIDIA CUDA）
python subtitle_generator.py video.mp4 --device cuda
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频或音频文件路径 | 自动检测当前目录 |
| `--output, -o` | 输出 SRT 文件路径 | 输入文件同目录 |
| `--device, -d` | 推理设备 (`cpu` / `cuda` / `cuda:0` / `cuda:1`) | `cpu` |
| `--compute-type` | 计算精度 (`int8` / `float16` / `float32`) | `int8` |
| `--speakers` | 启用说话人分离 | 关闭 |
| `--hf-token` | HuggingFace Access Token | 环境变量或 `.hf_token` |
| `--save-token` | 将 Token 保存到本地文件 | 不保存 |

### 说话人分离配置

说话人分离功能需要 HuggingFace Token：

1. 注册 [huggingface.co](https://huggingface.co)
2. 访问 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 点击 "Agree and access repository"
3. 访问 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 点击 "Agree and access repository"
4. 在 [Settings/Tokens](https://huggingface.co/settings/tokens) 创建 Access Token

### 国内用户（网络优化）

脚本默认使用 HuggingFace 镜像 `hf-mirror.com`，无需额外配置。

## 输出示例

```
1
00:00:01,200 --> 00:00:04,500
大家好，欢迎收看本期视频

2
00:00:04,500 --> 00:00:08,300
[说话人1] 今天我们来聊聊人工智能的话题
```

启用说话人分离后，每行字幕前会标注说话人编号。

## GPU 加速

- **推荐**: NVIDIA GPU + CUDA 11.8+，驱动 >= 522.06
- 使用 `--device cuda --compute-type float16` 可大幅加速
- 显存建议 >= 6GB（large-v3 模型约占用 3-4GB）

## 依赖

- [whisperx](https://github.com/m-bain/whisperX) >= 3.1.0 — 语音识别 + 对齐 + 说话人分离流水线
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) >= 3.1.0 — 说话人分离（可选）
- [torch](https://pytorch.org) >= 2.0.0
- [ffmpeg](https://ffmpeg.org) — 音频提取

## 许可证

MIT
