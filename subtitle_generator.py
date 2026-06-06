#!/usr/bin/env python3
"""
视频字幕生成脚本 (WhisperX + pyannote.audio)
- 识别语言: 中文、英文（自动检测）
- 功能: 语音识别 + 词级对齐 + 说话人分离
- 输出格式: SRT 字幕文件（带时间戳和说话人标签）

依赖:
    pip install whisperx pyannote.audio

模型（首次运行自动下载）:
    - faster-whisper large-v3: 多语言语音识别
    - wav2vec2: 词级强制对齐（中/英）
    - pyannote/speaker-diarization-3.1: 说话人分离

HuggingFace Token:
    pyannote 模型需要 HuggingFace 账号并接受使用协议。
    访问 https://huggingface.co/pyannote/speaker-diarization-3.1 点击 "Agree and access repository"
    访问 https://huggingface.co/pyannote/segmentation-3.0 点击 "Agree and access repository"
    然后在 https://huggingface.co/settings/tokens 创建 Access Token。

用法:
    python subtitle_generator.py                           # 自动检测当前目录视频
    python subtitle_generator.py input_video.mp4
    python subtitle_generator.py input_video.mp4 --speakers --device cuda
    python subtitle_generator.py input_video.mp4 --speakers --hf-token hf_xxxx
"""

import argparse
import os
import re
import subprocess
import sys

# 国内优先使用镜像，避免 HuggingFace Hub 连接超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
import tempfile
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

# pyannote 的 torchcodec 警告不相关（我们使用 ffmpeg CLI 提取音频）
warnings.filterwarnings("ignore", message=".*torchcodec.*")

# ========== 常量 ==========
HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_token")


# ========== 进度工具 ==========
class ProgressSpinner:
    """后台线程，每隔 interval 秒打印一个字符，表示程序仍在运行"""

    def __init__(self, msg="  处理中", interval=3.0):
        self.msg = msg
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._dots = 0

    def _run(self):
        while not self._stop.wait(self.interval):
            self._dots += 1
            dots = "." * ((self._dots % 4) + 1)
            print(f"\r{self.msg}{dots}   ", end="", flush=True)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        print("\r" + " " * 60 + "\r", end="", flush=True)


def log_step(msg: str):
    """带时间戳的步骤日志"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ========== 环境检查 ==========
def check_dependencies(speakers: bool = False) -> bool:
    missing = []

    try:
        import whisperx  # noqa: F401
    except ImportError:
        missing.append("whisperx (pip install whisperx)")

    if speakers:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            missing.append("pyannote.audio (pip install pyannote.audio)")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg")

    if missing:
        print("\n缺少以下依赖，请先安装:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True


def get_hf_token(cli_token: Optional[str] = None) -> str:
    """获取 HuggingFace token"""
    if cli_token:
        return cli_token
    if os.environ.get(HF_TOKEN_ENV):
        return os.environ[HF_TOKEN_ENV]
    if os.path.exists(HF_TOKEN_FILE):
        with open(HF_TOKEN_FILE, "r") as f:
            token = f.read().strip()
            if token:
                return token
    return ""


def save_hf_token(token: str):
    with open(HF_TOKEN_FILE, "w") as f:
        f.write(token.strip())
    os.chmod(HF_TOKEN_FILE, 0o600)
    print(f"    Token 已保存到 {HF_TOKEN_FILE}")


# ========== 音频提取 ==========
def extract_audio(video_path: str) -> str:
    log_step(f"[1/6] 正在从视频提取音频: {os.path.basename(video_path)}")

    tmp_dir = tempfile.gettempdir()
    audio_path = os.path.join(tmp_dir, f"whisperx_audio_{os.getpid()}.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取音频失败:\n{result.stderr}")

        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"    音频已提取 ({file_size:.1f} MB)")
        return audio_path
    except Exception as e:
        print(f"    错误: {e}")
        sys.exit(1)


# ========== 模型加载与推理 ==========
def load_whisper_model(device: str, compute_type: str):
    """加载 WhisperX 模型 (faster-whisper backend)"""
    import whisperx

    log_step(f"[2/6] 正在加载 Whisper 模型 (large-v3, device={device}, compute={compute_type})...")
    print("    (首次运行需下载模型, 约 3GB)")

    spinner = ProgressSpinner("  加载模型中", interval=3.0).start()
    try:
        model = whisperx.load_model(
            "large-v3",
            device=device,
            compute_type=compute_type,
            language=None,
        )
    finally:
        spinner.stop()
    print("    Whisper 模型加载完成!")
    return model


def transcribe_with_vad(model, audio, device: str):
    """使用 VAD 分段 + Whisper 转录"""
    import whisperx

    log_step("[3/6] 正在转录 (语音活动检测 + 语音识别)...")

    spinner = ProgressSpinner("  转录中", interval=3.0).start()
    try:
        result = model.transcribe(
            audio,
            batch_size=16,
            language=None,
        )
    finally:
        spinner.stop()

    detected_lang = result.get("language", "unknown")
    segments = result.get("segments", [])
    print(f"    检测到语言: {detected_lang}, 共 {len(segments)} 个语音段")

    # 实时显示识别出的前几段文本
    if segments:
        print("    --- 识别内容预览 ---")
        for seg in segments[:5]:
            text = seg.get("text", "").strip()
            ts = seg.get("start", 0)
            if text:
                m, s = divmod(int(ts), 60)
                h, m = divmod(m, 60)
                print(f"    [{h:02d}:{m:02d}:{s:02d}] {text}")
        if len(segments) > 5:
            print(f"    ... 还有 {len(segments) - 5} 段")
        print("    -------------------")

    return result, detected_lang


def align_transcript(result: dict, audio, device: str, detected_lang: str):
    """词级强制对齐"""
    import whisperx

    log_step(f"[4/6] 正在词级对齐 (语言: {detected_lang})...")

    lang_map = {"zh": "zh", "en": "en"}
    lang_code = lang_map.get(detected_lang, detected_lang)

    try:
        spinner = ProgressSpinner("  对齐中", interval=3.0).start()
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=lang_code,
                device=device,
            )
            result_aligned = whisperx.align(
                result["segments"],
                model=align_model,
                align_model_metadata=align_metadata,
                audio=audio,
                device=device,
                interpolate_method="nearest",
            )
        finally:
            spinner.stop()
        print(f"    对齐完成, {len(result_aligned.get('segments', []))} 个词级片段")
        return result_aligned
    except Exception as e:
        print(f"    对齐失败 ({e})，使用原始时间戳")
        return result


def diarize_speakers(result: dict, audio, hf_token: str, device: str):
    """使用 pyannote.audio 进行说话人分离"""
    import whisperx

    log_step("[5/6] 正在执行说话人分离 (pyannote/speaker-diarization-3.1)...")

    try:
        spinner = ProgressSpinner("  分离说话人中", interval=3.0).start()
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=10)
        finally:
            spinner.stop()

        result = whisperx.assign_word_speakers(diarize_segments, result)

        speakers = set()
        for seg in result.get("segments", []):
            speaker = seg.get("speaker")
            if speaker is not None:
                speakers.add(speaker)
        print(f"    检测到 {len(speakers)} 位说话人")
        if speakers:
            spk_list = sorted(speakers)
            print(f"    说话人: {', '.join(spk_list)}")
        return result
    except Exception as e:
        print(f"    说话人分离失败 ({e})，跳过")
        return result


# ========== 文本处理 ==========
def is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x3040 <= cp <= 0x309F) or
        (0x30A0 <= cp <= 0x30FF) or
        (0xAC00 <= cp <= 0xD7AF)
    )


def effective_length(text: str) -> float:
    length = 0.0
    for ch in text:
        if is_cjk(ch):
            length += 1.0
        elif ch.isspace():
            length += 0.25
        else:
            length += 0.5
    return length


def clean_text(text: str) -> str:
    """清理特殊标签和无效内容"""
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = text.replace('▁', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[[A-Z_]+\]', '', text)
    return text


def normalize_english_spacing(text: str) -> str:
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r"\s+'(\w+)", r"'\1", text)
    text = re.sub(r"\bn\s+'t\b", "n't", text)
    text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
    return text


def is_filler_phrase(text: str) -> bool:
    text_lower = text.strip().rstrip('.').rstrip('!').rstrip('?').lower()
    fillers = {'yeah', 'um', 'uh', 'er', 'hmm', 'ah', 'oh', 'well', 'so', 'okay', 'ok', 'yes', 'no'}
    return text_lower in fillers


def is_valid_subtitle(text: str) -> bool:
    has_content = bool(re.search(r'[a-zA-Z一-鿿]', text))
    return has_content and len(text.strip()) >= 2


def remove_stutter(text: str) -> str:
    words = text.split()
    if len(words) <= 1:
        return text

    def word_key(w):
        return w.strip('.,!?;:').lower()

    result = [words[0]]
    for i in range(1, len(words)):
        w = words[i]
        if word_key(w) == word_key(result[-1]) and len(word_key(w)) <= 5:
            continue
        if (i >= 2 and len(result) >= 2 and
            word_key(w) == word_key(result[-2]) and
            word_key(words[i-1]) == word_key(result[-1])):
            continue
        result.append(w)
    return ' '.join(result)


# ========== 字幕分组合并 ==========
def group_phrases_into_subtitles(phrases: List[dict]) -> List[dict]:
    if not phrases:
        return []

    subtitles = []
    buf_start = phrases[0]["start"]
    buf_end = phrases[0]["end"]
    buf_texts = [phrases[0]["text"]]
    buf_speaker = phrases[0].get("speaker", "SPEAKER_00")

    for i in range(1, len(phrases)):
        p = phrases[i]
        gap = p["start"] - buf_end
        combined_dur = p["end"] - buf_start
        combined_text = ' '.join(buf_texts + [p["text"]])
        combined_len = effective_length(combined_text)

        break_here = False
        if gap > 500:
            break_here = True
        elif combined_dur > 5000:
            break_here = True
        elif combined_len > 30:
            break_here = True

        if break_here:
            subtitles.append({
                "text": ' '.join(buf_texts),
                "start": buf_start,
                "end": buf_end,
                "speaker": buf_speaker,
            })
            buf_start = p["start"]
            buf_end = p["end"]
            buf_texts = [p["text"]]
            buf_speaker = p.get("speaker", "SPEAKER_00")
        else:
            buf_texts.append(p["text"])
            buf_end = p["end"]
            if p.get("speaker") != buf_speaker and len(buf_texts) > 1:
                speakers_in = [x.get("speaker", "SPEAKER_00") for x in
                               [{"speaker": buf_speaker}] * len(buf_texts[:-1]) + [{"speaker": p.get("speaker")}]]
                buf_speaker = max(set(speakers_in), key=speakers_in.count)
                # simpler: keep majority
                all_spks = [buf_speaker] * (len(buf_texts) - 1) + [p.get("speaker", "SPEAKER_00")]
                buf_speaker = max(set(all_spks), key=all_spks.count)

    if buf_texts:
        subtitles.append({
            "text": ' '.join(buf_texts),
            "start": buf_start,
            "end": buf_end,
            "speaker": buf_speaker,
        })

    return _fix_orphan_boundaries(subtitles)


def _fix_orphan_boundaries(subtitles: List[dict]) -> List[dict]:
    if len(subtitles) <= 1:
        return subtitles

    orphan_starts = {'and', 'or', 'but', 'into', 'with', 'as', 'by', 'to',
                     'of', 'in', 'on', 'at', 'for', 'that', 'which', 'the',
                     'more', 'it', 'is'}

    fixed = [subtitles[0]]
    for i in range(1, len(subtitles)):
        curr = subtitles[i]
        text = curr["text"].strip()
        first_word = text.split()[0].lower().rstrip('.').rstrip(',') if text.split() else ''

        if first_word in orphan_starts and len(text.split()) <= 5:
            prev = fixed[-1]
            combined = prev["text"] + ' ' + text
            combined_dur = curr["end"] - prev["start"]
            if effective_length(combined) <= 30 and combined_dur <= 6000:
                fixed[-1] = {
                    "text": combined,
                    "start": prev["start"],
                    "end": curr["end"],
                    "speaker": prev.get("speaker", "SPEAKER_00"),
                }
                continue
        fixed.append(curr)
    return fixed


def _remerge_short_subtitles(subtitles: List[dict]) -> List[dict]:
    if len(subtitles) <= 1:
        return subtitles

    result = []
    i = 0
    while i < len(subtitles):
        curr = subtitles[i]
        curr_dur = curr["end"] - curr["start"]
        curr_len = effective_length(curr["text"])

        if curr_dur < 1000 or curr_len < 10:
            merged = False
            if i + 1 < len(subtitles):
                next_seg = subtitles[i + 1]
                combined_dur = next_seg["end"] - curr["start"]
                combined_len = effective_length(curr["text"] + ' ' + next_seg["text"])
                gap = next_seg["start"] - curr["end"]

                if combined_dur <= 6000 and combined_len <= 32 and gap < 1500:
                    result.append({
                        "text": curr["text"] + ' ' + next_seg["text"],
                        "start": curr["start"],
                        "end": next_seg["end"],
                        "speaker": curr.get("speaker", "SPEAKER_00"),
                    })
                    i += 2
                    merged = True

            if not merged and result:
                prev = result[-1]
                combined_dur = curr["end"] - prev["start"]
                combined_len = effective_length(prev["text"] + ' ' + curr["text"])
                gap = curr["start"] - prev["end"]

                if combined_dur <= 6000 and combined_len <= 32 and gap < 1500:
                    result[-1] = {
                        "text": prev["text"] + ' ' + curr["text"],
                        "start": prev["start"],
                        "end": curr["end"],
                        "speaker": prev.get("speaker", "SPEAKER_00"),
                    }
                    i += 1
                    merged = True

            if not merged:
                result.append(curr)
                i += 1
        else:
            result.append(curr)
            i += 1

    return result


# ========== SRT 生成 ==========
def ms_to_srt_time(ms: float) -> str:
    ms_int = max(0, int(ms))
    hours = ms_int // 3600000
    minutes = (ms_int % 3600000) // 60000
    seconds = (ms_int % 60000) // 1000
    millis = ms_int % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def speaker_label(speaker_id: str, speaker_map: dict = None) -> str:
    """
    将 SPEAKER_00 映射为用户友好的标签。
    SPEAKER_00 → 说话人1, SPEAKER_01 → 说话人2, 等等
    """
    if speaker_map and speaker_id in speaker_map:
        return f"[说话人{speaker_map[speaker_id]}]"
    m = re.match(r'SPEAKER_(\d+)', speaker_id)
    if m:
        num = int(m.group(1)) + 1
        return f"[说话人{num}]"
    return f"[{speaker_id}]"


def generate_srt(segments: List[dict]) -> str:
    lines = []
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    # 收集所有不重复的 speaker，按首次出现顺序编号
    speaker_ids = []
    for seg in sorted_segs:
        spk = seg.get("speaker", "SPEAKER_00")
        if spk not in speaker_ids:
            speaker_ids.append(spk)
    speaker_map = {spk: str(i + 1) for i, spk in enumerate(speaker_ids)}

    for idx, seg in enumerate(sorted_segs):
        start = ms_to_srt_time(seg["start"])
        end = ms_to_srt_time(seg["end"])
        text = seg["text"]
        spk = seg.get("speaker", "SPEAKER_00")

        lines.append(str(idx + 1))
        lines.append(f"{start} --> {end}")
        if len(speaker_ids) > 1:
            lines.append(f"{speaker_label(spk, speaker_map)} {text}")
        else:
            lines.append(text)
        lines.append("")

    return "\n".join(lines)


# ========== 后处理 ==========
def postprocess_segments(segments_raw: List[dict], text_raw: str) -> List[dict]:
    """将 WhisperX 结果解析为标准片段格式"""
    phrases = []
    for seg in segments_raw:
        text = clean_text(seg.get("text", ""))
        text = text.strip()
        if not text or not is_valid_subtitle(text):
            continue
        if is_filler_phrase(text):
            continue

        start_ms = int(seg.get("start", 0) * 1000)
        end_ms = int(seg.get("end", 0) * 1000)

        phrases.append({
            "text": text,
            "start": start_ms,
            "end": end_ms,
            "speaker": seg.get("speaker", "SPEAKER_00"),
        })

    if not phrases:
        return []

    # 文本清理
    for p in phrases:
        p["text"] = clean_text(p["text"])
        p["text"] = remove_stutter(p["text"])
        p["text"] = normalize_english_spacing(p["text"])

    # 过滤
    phrases = [p for p in phrases
               if is_valid_subtitle(p["text"]) and not is_filler_phrase(p["text"])]

    if not phrases:
        return []

    # 分组为字幕行
    subtitles = group_phrases_into_subtitles(phrases)
    print(f"    短语分组后: {len(subtitles)} 个字幕行")

    # 合并过短字幕
    subtitles = _remerge_short_subtitles(subtitles)

    # 确保最小时长
    for i, seg in enumerate(subtitles):
        dur = seg["end"] - seg["start"]
        if dur < 1000:
            seg["end"] = seg["start"] + 1000
            if i + 1 < len(subtitles):
                next_start = subtitles[i + 1]["start"]
                if seg["end"] >= next_start:
                    seg["end"] = next_start - 50

    # 避免重叠
    for i in range(len(subtitles) - 1):
        if subtitles[i]["end"] >= subtitles[i + 1]["start"]:
            subtitles[i]["end"] = subtitles[i + 1]["start"] - 50

    # 二次清理
    for seg in subtitles:
        seg["text"] = remove_stutter(seg["text"])
        seg["text"] = normalize_english_spacing(seg["text"])

    return subtitles


# ========== 视频处理流水线 ==========
def is_video_file(file_path: str) -> bool:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    return Path(file_path).suffix.lower() in video_extensions


def process(input_path: str, output_path: str, device: str = "cpu",
            compute_type: str = "int8", speakers: bool = False,
            hf_token: str = "", save_token: bool = False):
    import whisperx

    audio_path = None
    try:
        if is_video_file(input_path):
            audio_path = extract_audio(input_path)
        else:
            audio_path = input_path
            log_step(f"[1/6] 使用音频文件: {os.path.basename(input_path)}")

        # 加载音频
        audio = whisperx.load_audio(audio_path)

        # 加载模型 & 转录
        model = load_whisper_model(device, compute_type)
        result, detected_lang = transcribe_with_vad(model, audio, device)

        del model
        torch.cuda.empty_cache()

        # 词级对齐
        result = align_transcript(result, audio, device, detected_lang)

        # 说话人分离
        if speakers:
            if not hf_token:
                print("\n    错误: 说话人分离需要 HuggingFace token!")
                print("    使用 --hf-token 参数或设置 HF_TOKEN 环境变量")
                print("    或创建 .hf_token 文件包含你的 token")
                sys.exit(1)
            if save_token and hf_token:
                save_hf_token(hf_token)
            result = diarize_speakers(result, audio, hf_token, device)

        # 后处理
        segments_raw = result.get("segments", [])
        if not segments_raw:
            print("    警告: 未识别到任何语音内容!")
            return

        print(f"\n    --- 后处理 {len(segments_raw)} 个原始片段 ---")
        subtitles = postprocess_segments(segments_raw, result.get("text", ""))

        if not subtitles:
            print("    警告: 无有效字幕内容!")
            return

        log_step(f"[6/6] 正在生成字幕文件: {os.path.basename(output_path)}")
        speakers_count = len(set(s.get("speaker", "SPEAKER_00") for s in subtitles))
        print(f"    共 {len(subtitles)} 个字幕片段, {speakers_count} 位说话人")

        # 文本清理
        for seg in subtitles:
            seg["text"] = clean_text(seg["text"])

        srt_content = generate_srt(subtitles)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"\n[DONE] 字幕生成完成!")
        print(f"   输出文件: {os.path.abspath(output_path)}")

        # 打印预览
        print("\n--- 字幕预览 (前8条) ---")
        preview_blocks = srt_content.split("\n\n")[:8]
        for p in preview_blocks:
            if p.strip():
                print(p.strip())
                print()

    finally:
        if audio_path and is_video_file(input_path) and os.path.exists(audio_path):
            os.remove(audio_path)
            print("    临时音频文件已清理。")


# ========== 命令行入口 ==========
def find_video_in_directory(directory: str) -> Optional[str]:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            return str(file)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="视频字幕生成脚本 - WhisperX + pyannote.audio (中英双语, 说话人分离)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python subtitle_generator.py                              # 自动检测当前目录视频
    python subtitle_generator.py video.mp4                    # 指定视频文件
    python subtitle_generator.py video.mp4 --speakers         # 启用说话人分离
    python subtitle_generator.py video.mp4 --speakers --hf-token hf_xxxx
    python subtitle_generator.py video.mp4 --device cuda      # 使用 GPU
    python subtitle_generator.py audio.wav --output out.srt
        """,
    )

    parser.add_argument("input", nargs="?", default=None,
                        help="输入视频或音频文件 (不填则自动检测当前文件夹中的视频)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出 SRT 字幕文件路径 (默认: 与输入同名 .srt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
                        help="推理设备 (默认: cpu)")
    parser.add_argument("--compute-type", default="int8",
                        choices=["int8", "float16", "float32", "int8_float16", "int8_float32"],
                        help="Whisper 计算精度 (默认: int8)")
    parser.add_argument("--speakers", action="store_true",
                        help="启用说话人分离 (pyannote.audio)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace Access Token (用于 pyannote 模型)")
    parser.add_argument("--save-token", action="store_true",
                        help="将 HF token 保存到本地 .hf_token 文件")

    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = find_video_in_directory(os.getcwd())
        if not input_path:
            print("错误: 当前目录未找到视频文件")
            print("请将视频放到当前目录，或作为参数指定")
            sys.exit(1)
        print(f"使用视频文件: {input_path}\n")

    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        # 输出到输入文件同目录
        input_dir = os.path.dirname(os.path.abspath(input_path))
        output_path = os.path.join(input_dir, f"{Path(input_path).stem}.srt")

    if not check_dependencies(args.speakers):
        sys.exit(1)

    hf_token = get_hf_token(args.hf_token)
    process(input_path, output_path, args.device, args.compute_type,
            args.speakers, hf_token, args.save_token)


if __name__ == "__main__":
    main()
