#!/usr/bin/env python3
"""
run_full_meeting.py — Golden Config + Pyannote Diarization
=========================================================
Chạy pipeline đầy đủ trên audio dài:
  1. Tiền xử lý âm thanh (pre-emphasis + bandpass EQ)
  2. Pyannote diarization → xác định từng người nói
  3. Faster-Whisper large-v3-turbo (Golden Config) → transcribe từng đoạn
  4. Xuất 2 file:
     - <audio>_transcript.txt  : toàn bộ văn bản theo thời gian
     - <audio>_diarized.txt    : văn bản gán nhãn người nói

Usage:
  conda run -n vmeeting python serving/run_full_meeting.py
  conda run -n vmeeting python serving/run_full_meeting.py --audio path/to/audio.m4a
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
CT2_MODEL   = os.path.expanduser("~/.cache/whisper-ct2/large-v3-turbo")

# ── Golden Config ──────────────────────────────────────────────────────────────
INITIAL_PROMPT = (
    "Cuộc họp nội bộ Viettel Netmind về xây dựng mô hình nền tảng ngôn ngữ "
    "tiết kiệm năng lượng, dự án Power Saving, hệ thống Agentic N8N."
)
HOTWORDS = "Netmind PowerSaving Power Saving OpenAI N8N LLM Ericsson Agentic Transformer"

WHISPER_KWARGS = dict(
    language="vi",
    task="transcribe",
    beam_size=5,
    best_of=5,
    no_speech_threshold=0.45,
    compression_ratio_threshold=2.0,
    log_prob_threshold=-1.0,
    hallucination_silence_threshold=2.0,
    condition_on_previous_text=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vad_filter=False,
    initial_prompt=INITIAL_PROMPT,
    hotwords=HOTWORDS,
)

# ── Hallucination / hotword-leak patterns ──────────────────────────────────────
HALLUCINATIONS = [
    "hãy subscribe", "hãy đăng ký kênh", "đăng ký kênh", "ủng hộ kênh",
    "lalaschool", "ghiền mì gõ", "để không bỏ lỡ", "cảm ơn các bạn đã theo dõi",
    "hẹn gặp lại", "tạm biệt các bạn", "like và subscribe", "nhấn like",
    "chuông thông báo", "bạn đã xem video", "viết phụ đề bởi",
]
HOTWORD_LEAKS = [
    "N8N LLM Ericsson Agentic Transformer",
    "PowerSaving Power Saving OpenAI N8N",
    "Netmind PowerSaving Power Saving",
    "LLM Ericsson Agentic Transformer",
    "LLM Ericsson Agent",
    "N1N LLM Ericsson",
]


# ==============================================================================
# Logger
# ==============================================================================

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()


# ==============================================================================
# Audio utils
# ==============================================================================

def load_audio(path: str) -> np.ndarray:
    """Load bất kỳ định dạng nào → 16kHz mono float32."""
    import soundfile as sf
    import subprocess
    import tempfile

    try:
        import librosa
        audio, sr = sf.read(path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio.astype(np.float32)
    except Exception:
        pass

    # Fallback: ffmpeg decode
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ffmpeg = "/home/h2n/anaconda3/envs/vmeeting/bin/ffmpeg"
        if not os.path.exists(ffmpeg):
            ffmpeg = "ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-i", path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_path],
            check=True, capture_output=True,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
        return audio.astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def enhance_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Pre-emphasis + Bandpass EQ (100–8000 Hz) + peak norm. KHÔNG dùng AI denoise."""
    from scipy import signal

    audio = audio.copy().astype(np.float32)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # RMS norm → -18 dBFS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-8:
        audio = audio * (10 ** (-18.0 / 20.0) / rms)
    audio = np.clip(audio, -1.0, 1.0)

    # Bandpass: tần số người nói 100–8000 Hz
    nyq = sr / 2.0
    try:
        sos = signal.butter(5, [100.0 / nyq, 8000.0 / nyq], btype="band", output="sos")
        audio = signal.sosfilt(sos, audio)
    except Exception:
        pass

    # Peak norm → 0.95
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


# ==============================================================================
# Post-processing
# ==============================================================================

def post_process(text: str) -> str:
    import re

    # Hotword leak từ beam search
    for pat in HOTWORD_LEAKS:
        if pat.lower() in text.lower():
            text = text[: text.lower().find(pat.lower())].strip()

    # Regex: chuỗi hotwords rơi vào cuối câu
    text = re.sub(
        r'\b(N8N|N1N|LLM|Ericsson|Agentic|Transformer|PowerSaving|Netmind)'
        r'(\s+(N8N|N1N|LLM|Ericsson|Agentic|Transformer|PowerSaving|Netmind|Agent|C\d))+\s*$',
        "", text, flags=re.IGNORECASE,
    ).strip()

    # Hallucination cắt đuôi
    lower = text.lower()
    for h in HALLUCINATIONS:
        if h in lower:
            text = text[: lower.find(h)].strip()
            lower = text.lower()

    # Lặp từ vô tận
    words = text.split()
    if len(words) > 5:
        clean = []
        for w in words:
            if len(clean) >= 3 and clean[-1] == w == clean[-2] == clean[-3]:
                continue
            clean.append(w)
        text = " ".join(clean)

    return text.strip()


# ==============================================================================
# VAD (Silero)
# ==============================================================================

def get_vad_chunks(audio: np.ndarray, duration: float) -> tuple[list, list]:
    """
    Trả về (chunks, metas) đã cắt theo VAD, mỗi chunk ≤ 29s.
    Dùng cho transcript-only (không diarization).
    """
    vad_path = os.path.join(SERVING_DIR, "models", "silero-vad")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir=vad_path, model="silero_vad", source="local", trust_repo=True,
    )
    if torch.cuda.is_available():
        vad_model = vad_model.cuda()

    audio_t = torch.from_numpy(audio)
    if torch.cuda.is_available():
        audio_t = audio_t.cuda()

    with torch.no_grad():
        raw_ts = vad_utils[0](
            audio_t, vad_model, sampling_rate=16000, return_seconds=True,
            threshold=0.35, min_speech_duration_ms=300, min_silence_duration_ms=150,
        )

    del vad_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    segs, pad = [], 0.40
    if raw_ts:
        cs, ce = raw_ts[0]["start"], raw_ts[0]["end"]
        for s in raw_ts[1:]:
            if s["start"] - ce <= 1.5 and s["end"] - cs <= 29.0:
                ce = s["end"]
            else:
                segs.append({"start": max(0.0, cs - pad), "end": min(duration, ce + pad)})
                cs, ce = s["start"], s["end"]
        segs.append({"start": max(0.0, cs - pad), "end": min(duration, ce + pad)})

    MAX_C, MIN_RMS = 29 * 16000, 0.008
    chunks, meta = [], []
    for seg in segs:
        s, e = int(seg["start"] * 16000), int(seg["end"] * 16000)
        for j in range(0, e - s, MAX_C):
            c = audio[s + j : s + j + MAX_C]
            if len(c) / 16000 < 0.2:
                continue
            if np.sqrt(np.mean(c.astype(np.float64) ** 2)) < MIN_RMS:
                continue
            chunks.append(c)
            meta.append({
                "start": seg["start"] + j / 16000,
                "end": seg["start"] + (j + len(c)) / 16000,
            })

    return chunks, meta


# ==============================================================================
# Pyannote diarization
# ==============================================================================

def _patch_pyannote_config(config_path: str) -> str:
    """
    Đọc config.yaml và thay thế bất kỳ đường dẫn tuyệt đối nào của segmentation/embedding
    bằng đường dẫn tương đối cạnh config.yaml. Ghi ra file tạm, trả về path đó.
    """
    import tempfile, re

    models_dir = os.path.join(SERVING_DIR, "models", "pyannote")

    with open(config_path, "r") as f:
        content = f.read()

    # Thay thế bất kỳ path tuyệt đối nào kết thúc bằng pytorch_model.bin
    def replace_path(m):
        original = m.group(0)
        # Lấy tên thư mục model (segmentation-3.0 / wespeaker-...)
        parts = original.replace("\\", "/").split("/")
        # Tìm 2 phần cuối: <model_dir>/pytorch_model.bin
        if len(parts) >= 2:
            model_subdir = parts[-2]
            filename     = parts[-1]
            new_path = os.path.join(models_dir, model_subdir, filename)
            if os.path.exists(new_path):
                return new_path
        return original

    patched = re.sub(r"[^\s:]+pytorch_model\.bin", replace_path, content)

    # Ghi ra file tạm
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=os.path.dirname(config_path)
    )
    tmp.write(patched)
    tmp.close()
    return tmp.name


def run_diarization(audio: np.ndarray) -> list[tuple[float, float, str]]:
    """
    Chạy Pyannote 3.1 → list[(start, end, speaker_label)].
    Speaker label được re-index theo thứ tự xuất hiện (Speaker 1, Speaker 2, …).
    """
    from pyannote.audio import Pipeline

    config_path = os.path.join(
        SERVING_DIR, "models", "pyannote", "speaker-diarization-3.1", "config.yaml"
    )
    # Patch paths (config.yaml có thể chứa absolute path từ máy khác)
    patched_config = _patch_pyannote_config(config_path)
    print(f"🔍 Đang load Pyannote 3.1 Pipeline (config: {patched_config})...")
    try:
        pipeline = Pipeline.from_pretrained(patched_config)
    finally:
        if patched_config != config_path and os.path.exists(patched_config):
            os.unlink(patched_config)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        if hasattr(pipeline, "segmentation_batch_size"):
            pipeline.segmentation_batch_size = 4
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = 4

    print("🎙️  Đang diarize (có thể mất vài phút với audio dài)...")
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diarization = pipeline({"waveform": waveform, "sample_rate": 16000})

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse annotation
    raw = []
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw.append((turn.start, turn.end, speaker))

    if not raw:
        return []

    # Re-index speakers theo thứ tự xuất hiện đầu tiên
    raw.sort(key=lambda x: x[0])
    order: dict[str, int] = {}
    timeline = []
    for start, end, spk in raw:
        if spk not in order:
            order[spk] = len(order) + 1
        timeline.append((start, end, f"Speaker {order[spk]}"))

    return timeline


def merge_diarization(timeline: list, max_gap: float = 1.5, max_dur: float = 25.0) -> list:
    """Merge các đoạn liên tiếp cùng speaker, gap ≤ max_gap và tổng ≤ max_dur."""
    if not timeline:
        return []

    merged = []
    cs, ce, cspk = timeline[0]
    for s, e, spk in timeline[1:]:
        if spk == cspk and s - ce <= max_gap and (e - cs) <= max_dur:
            ce = max(ce, e)
        else:
            if ce - cs >= 0.3:
                merged.append((cs, ce, cspk))
            cs, ce, cspk = s, e, spk
    if ce - cs >= 0.3:
        merged.append((cs, ce, cspk))
    return merged


# ==============================================================================
# Transcription
# ==============================================================================

def transcribe_chunks(model, chunks: list, metas: list) -> list[dict]:
    """
    Transcribe list chunks, trả về list[{"start", "end", "text"}].
    """
    results = []
    for chunk, meta in zip(chunks, metas):
        fw_segs, _ = model.transcribe(chunk.astype(np.float32), **WHISPER_KWARGS)
        raw = " ".join(s.text.strip() for s in fw_segs).strip()
        text = post_process(raw)
        if text:
            results.append({"start": meta["start"], "end": meta["end"], "text": text})
    return results


def transcribe_segment(model, audio: np.ndarray, start: float, end: float) -> str:
    """Transcribe một đoạn audio (max 29s)."""
    fw_segs, _ = model.transcribe(audio.astype(np.float32), **WHISPER_KWARGS)
    raw = " ".join(s.text.strip() for s in fw_segs).strip()
    return post_process(raw)


# ==============================================================================
# Main
# ==============================================================================

def main():
    default_audio = os.path.join(os.path.dirname(SERVING_DIR), "audio_full.m4a")
    parser = argparse.ArgumentParser(description="Golden Whisper + Diarization pipeline")
    parser.add_argument("--audio", default=default_audio, help="Path to audio file")
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Bỏ qua diarization, chỉ chạy transcript thuần",
    )
    args = parser.parse_args()

    # Setup log
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(SERVING_DIR, f"run_full_meeting_{ts}.log")
    sys.stdout = Logger(log_path)
    print(f"📄 Log: {log_path}")

    if not os.path.exists(args.audio):
        print(f"❌ Không tìm thấy audio: {args.audio}")
        return

    stem = os.path.splitext(os.path.basename(args.audio))[0]
    out_dir = os.path.dirname(args.audio)
    transcript_path = os.path.join(out_dir, f"{stem}_transcript.txt")
    diarized_path   = os.path.join(out_dir, f"{stem}_diarized.txt")

    print("\n" + "=" * 80)
    print("🌟 GOLDEN WHISPER + DIARIZATION — full meeting pipeline")
    print(f"   Audio  : {args.audio}")
    print(f"   Model  : large-v3-turbo (float16 / CTranslate2)")
    print(f"   Config : beam=5 | no_speech_thr=0.45 | temp=[0..1] | hotwords ON")
    print("=" * 80 + "\n")

    # ── 1. Load & enhance ─────────────────────────────────────────────────────
    print("🎵 [1/4] Load & enhance audio...")
    t0 = time.time()
    audio = load_audio(args.audio)
    duration = len(audio) / 16000.0
    audio = enhance_audio(audio)
    print(f"   ✅ {duration:.1f}s  ({time.time()-t0:.1f}s)\n")

    # ── 2. Load Whisper ────────────────────────────────────────────────────────
    print("🚀 [2/4] Load Faster-Whisper large-v3-turbo...")
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        CT2_MODEL, device=device,
        compute_type="float16" if device == "cuda" else "int8",
        num_workers=2,
    )
    print(f"   ✅ Device: {device.upper()}\n")

    # ── 3a. Transcript thuần (VAD) ────────────────────────────────────────────
    print("📡 [3/4] Silero VAD + Transcribe...")
    chunks, metas = get_vad_chunks(audio, duration)
    print(f"   → {len(chunks)} chunks\n")

    print("🎙️  Transcribing...")
    t1 = time.time()
    transcript_segs = transcribe_chunks(model, chunks, metas)
    t_elapsed = time.time() - t1
    print(f"   ✅ {t_elapsed:.1f}s  (RTF={t_elapsed/duration:.2f}x)\n")

    # In kết quả transcript
    print("=" * 80)
    print("📝 TRANSCRIPT (theo thời gian):")
    print("=" * 80)
    for seg in transcript_segs:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]  {seg['text']}")

    full_text = " ".join(s["text"] for s in transcript_segs)
    print("\n" + "=" * 80)
    print(f"📊 Tổng: {len(transcript_segs)} segments | {len(full_text)} chars | {t_elapsed:.1f}s STT")
    print("=" * 80)

    # Ghi file transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript — {args.audio}\n")
        f.write(f"# Audio duration: {duration:.1f}s | STT time: {t_elapsed:.1f}s | RTF: {t_elapsed/duration:.2f}x\n\n")
        for seg in transcript_segs:
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]  {seg['text']}\n")
        f.write(f"\n\n--- FULL TEXT ---\n{full_text}\n")
    print(f"\n💾 Transcript → {transcript_path}")

    # ── 3b. Diarization ───────────────────────────────────────────────────────
    if args.no_diarize:
        print("\n⚠️  --no-diarize: bỏ qua diarization.")
        return

    print("\n" + "=" * 80)
    print("👥 [4/4] Pyannote Diarization...")
    print("=" * 80)
    t2 = time.time()

    raw_timeline = run_diarization(audio)
    if not raw_timeline:
        print("⚠️  Pyannote không phát hiện giọng nói nào.")
        return

    timeline = merge_diarization(raw_timeline)
    speakers = sorted({spk for _, _, spk in timeline})
    print(f"   ✅ {len(speakers)} người nói | {len(timeline)} segments | {time.time()-t2:.1f}s\n")

    # Transcribe từng speaker segment
    print("🎙️  Transcribing speaker segments...")
    t3 = time.time()
    MAX_C = 29 * 16000
    diarized_segs = []

    for start, end, spk in timeline:
        s_idx = max(0, int((start - 0.15) * 16000))
        e_idx = min(len(audio), int((end + 0.15) * 16000))
        seg_audio = audio[s_idx:e_idx]
        seg_dur = len(seg_audio) / 16000.0

        if seg_dur < 0.2:
            continue

        # Nếu đoạn > 29s thì cắt nhỏ
        if len(seg_audio) > MAX_C:
            parts = []
            for j in range(0, len(seg_audio), MAX_C):
                part = seg_audio[j : j + MAX_C]
                if len(part) / 16000 < 0.2:
                    continue
                text = transcribe_segment(model, part, start + j / 16000, end)
                if text:
                    parts.append(text)
            combined = " ".join(parts)
        else:
            combined = transcribe_segment(model, seg_audio, start, end)

        if combined:
            diarized_segs.append({"start": start, "end": end, "speaker": spk, "text": combined})
            print(f"[{start:.1f}s - {end:.1f}s] [{spk}]: {combined}")

    print(f"\n   ✅ {time.time()-t3:.1f}s\n")

    # ── Output diarized ───────────────────────────────────────────────────────
    print("=" * 80)
    print("📝 DIARIZED TRANSCRIPT (người nói + thời gian):")
    print("=" * 80)
    for seg in diarized_segs:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] [{seg['speaker']}]: {seg['text']}")

    # Ghi file diarized
    with open(diarized_path, "w", encoding="utf-8") as f:
        f.write(f"# Diarized Transcript — {args.audio}\n")
        f.write(f"# Audio: {duration:.1f}s | Speakers: {len(speakers)}\n\n")
        for seg in diarized_segs:
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] [{seg['speaker']}]: {seg['text']}\n")

        # Full text theo từng người
        f.write("\n\n--- FULL TEXT BY SPEAKER ---\n")
        for spk in speakers:
            spk_text = " ".join(s["text"] for s in diarized_segs if s["speaker"] == spk)
            f.write(f"\n[{spk}]:\n{spk_text}\n")

    print(f"\n💾 Diarized → {diarized_path}")

    total = time.time() - t0
    print(f"\n✅ Xong! Tổng thời gian: {total:.1f}s (audio {duration:.1f}s, RTF total={total/duration:.2f}x)")
    print(f"   📄 {transcript_path}")
    print(f"   📄 {diarized_path}")


if __name__ == "__main__":
    main()
