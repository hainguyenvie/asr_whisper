import os
import sys

# ==========================================
# OFFLINE MODE - Chạy hoàn toàn local, không download
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# === FIX MALLOC(): UNALIGNED TCACHE CHUNK ===
# Ngăn chặn các thư viện C/C++ dưới gầm ngầm (OpenMP, MKL) đẻ hàng chục ngàn Threads
# Khi FastAPI gọi Multi-Workers, sự tranh chấp Thread C++ sẽ gây OOM hoặc Vỡ RAM (malloc crash).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import time
import io
import tempfile
import numpy as np
import soundfile as sf
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from scipy import signal
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ==========================================
# SETUP & UTILS
# ==========================================

app = FastAPI(title="Moonshine STT Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_audio_robust(file_source):
    """
    Robust audio loader that handles BytesIO or paths.
    Falls back to librosa/ffmpeg if soundfile fails (e.g. WebM).
    Returns: (audio_np_array, sample_rate)
    """
    if hasattr(file_source, 'seek'):
        file_source.seek(0)

    try:
        return sf.read(file_source, dtype='float32')
    except Exception as e:
        temp_path = None
        created_temp = False

        try:
            if hasattr(file_source, 'read'):
                file_source.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(file_source.read())
                    temp_path = tmp.name
                    created_temp = True
            else:
                temp_path = file_source

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import librosa
                audio, sr = librosa.load(temp_path, sr=None, mono=False)

            if len(audio.shape) > 1:
                audio = audio.T

            return audio, sr

        except Exception as e2:
            print(f"❌ Robust load failed: {e2}")
            raise e2
        finally:
            if created_temp and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


def enhance_audio_for_asr(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    D_LITE Pipeline: RMS normalization + Bandpass filter + Peak normalization
    """
    if len(audio) == 0:
        return audio

    audio = audio.astype(np.float32)

    # 1. RMS Normalization → target -20 dBFS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-8:
        audio = audio * (0.1 / rms)
        audio = np.clip(audio, -1.0, 1.0)

    # 2. Bandpass filter: 80 Hz – 7500 Hz
    nyq = sr / 2.0
    try:
        sos = signal.butter(4, [80.0 / nyq, 7500.0 / nyq], btype='band', output='sos')
        audio = signal.sosfilt(sos, audio)
    except Exception:
        pass

    # 3. Peak normalization → [-0.95, 0.95]
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


# ==========================================
# MOONSHINE ENGINE
# ==========================================

class MoonshineEngine:
    """
    Moonshine ASR với Silero VAD tích hợp.

    - diarization=false: Chỉ transcribe với VAD (live transcripts)
    - diarization=true:  Transcribe với Pyannote diarization (final/saved transcripts)
    """

    MAX_TOKENS_PER_SEC = 13.0
    MAX_CHUNK_SEC = 30.0

    def __init__(self, model_id=None):
        if model_id is None:
            self.model_id = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moonshine-base-vi")
        else:
            self.model_id = model_id
        self.processor = None
        self.model = None
        self.vad_model = None
        self.vad_utils = None
        self.loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def load(self):
        if self.loaded:
            return
        print(f"🚀 Loading Moonshine + Silero VAD on {self.device.upper()} ({self.torch_dtype})...")
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        # 1. Moonshine ASR model (OFFLINE MODE)
        self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
        self.model = MoonshineForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

        # 2. Silero VAD (Loaded from local copy)
        print("  📡 Loading Silero VAD...")
        try:
            vad_local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "silero-vad")
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir=vad_local_path,
                model="silero_vad",
                source="local",
                force_reload=False,
                trust_repo=True,
            )
            self.vad_model = self.vad_model.to(self.device)
            print("  ✅ Silero VAD loaded")
        except Exception as e:
            print(f"  ⚠️ Silero VAD load failed ({e}), will transcribe without VAD")
            self.vad_model = None

        self.loaded = True
        print(f"✅ MoonshineEngine ready")

    def unload(self):
        if self.loaded:
            print("🛑 Unloading Moonshine + VAD...")
            del self.model, self.processor
            if self.vad_model is not None:
                del self.vad_model
            import gc
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            self.loaded = False

    def _get_speech_segments(self, audio: np.ndarray, duration_sec: float) -> list:
        """Silero VAD với smart packing."""
        if self.vad_model is None:
            return [{"start": 0.0, "end": duration_sec}]

        get_speech_timestamps = self.vad_utils[0]
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        if self.device == "cuda":
            audio_tensor = audio_tensor.to(self.device)

        with moonshine_lock:
            with torch.no_grad():
                raw_timestamps = get_speech_timestamps(
                    audio_tensor,
                    self.vad_model,
                    sampling_rate=16000,
                    return_seconds=True,
                    threshold=0.5,
                    min_speech_duration_ms=500,
                    min_silence_duration_ms=200,
                )

        if not raw_timestamps:
            print("  ⚠️ VAD: không phát hiện voice")
            return []

        # Smart packing
        merged = []
        curr_start = raw_timestamps[0]["start"]
        curr_end = raw_timestamps[0]["end"]

        for seg in raw_timestamps[1:]:
            gap = seg["start"] - curr_end
            if gap <= 2.0 and (seg["end"] - curr_start <= 29.0):
                curr_end = seg["end"]
            else:
                merged.append({
                    "start": max(0.0, curr_start - 0.3),
                    "end": min(duration_sec, curr_end + 0.3)
                })
                curr_start = seg["start"]
                curr_end = seg["end"]

        merged.append({
            "start": max(0.0, curr_start - 0.3),
            "end": min(duration_sec, curr_end + 0.3)
        })

        print(f"  🎙️ VAD: {len(raw_timestamps)} raw → {len(merged)} smartly packed chunks")
        return merged

    def _transcribe_segment(self, audio_seg: np.ndarray) -> str:
        """Transcribe một numpy array audio sử dụng Moonshine."""
        if len(audio_seg) == 0:
            return ""

        inputs = self.processor(
            audio_seg,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device, self.torch_dtype)

        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            dur_sec = (inputs.attention_mask.sum(dim=-1).float().max().item()) / 16000.0
        else:
            dur_sec = len(audio_seg) / 16000.0

        max_new_tokens = max(10, int(dur_sec * 35.0))

        import warnings
        with moonshine_lock:  # Dùng riêng Lock để không bị Diarization (Pyannote) block
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*exceeded the model's predefined maximum length.*")
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    max_length=4096, # FIX TCACHE CHUNK DETECTED: Phá vỡ conflict giữa max_length và max_new_tokens gây tràn bộ nhớ C++!
                    do_sample=False,
                    num_beams=1,
                )

                return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _transcribe_batch(self, audio_segs: list[np.ndarray], batch_size: int = 8) -> list[str]:
        """Transcribe batch of audio segments to utilize GPU efficiently."""
        if not audio_segs:
            return []
        
        results = []
        for i in range(0, len(audio_segs), batch_size):
            batch = audio_segs[i:i+batch_size]
            inputs = self.processor(
                batch,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device, self.torch_dtype)
            
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                dur_sec = (inputs.attention_mask.sum(dim=-1).float().max().item()) / 16000.0
            else:
                dur_sec = max(len(a) for a in batch) / 16000.0
            
            max_new_tokens = max(10, int(dur_sec * 35.0))
            
            import warnings
            with moonshine_lock:
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*exceeded the model's predefined maximum length.*")
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            max_length=4096, # Ngăn chặn OOB memory overwrite
                            do_sample=False,
                            num_beams=1,
                        )
                    except Exception as e:
                        print(f"CUDA Error in generate batch: {e}")
                        continue
                
                decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                results.extend([d.strip() for d in decoded])
            
        return results

    def transcribe(self, audio_source) -> dict:
        """Transcribe với VAD (cho live transcripts)."""
        if not self.loaded:
            self.load()

        t_start = time.time()

        # Load audio → 16kHz mono float32
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            audio, sr = load_audio_robust(audio_source)
        else:
            audio, sr = load_audio_robust(io.BytesIO(audio_source)) if isinstance(audio_source, bytes) else load_audio_robust(audio_source)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)

        duration_sec = len(audio) / 16000.0
        print(f"🎵 Moonshine: {duration_sec:.2f}s audio")

        # Audio enhancement
        audio = enhance_audio_for_asr(audio, sr=16000)

        # VAD
        speech_segs = self._get_speech_segments(audio, duration_sec)

        HALLUCINATIONS = [
            # YouTube / subscribe hallucinations
            "hãy subscribe cho kênh",
            "hãy đăng ký kênh",
            "đăng ký kênh để ủng hộ",
            "ủng hộ kênh của mình",
            "like và subscribe",
            "nhấn like và đăng ký",
            "để không bỏ lỡ những video",
            "đừng quên đăng ký",
            "chuông thông báo",
            # Common Vietnamese subtitle hallucinations
            "ghiền mì gõ",
            "bạn đã xem video",
            "viết phụ đề bởi",
            "người dịch:",
            "cảm ơn các bạn",
            "hẹn gặp lại các bạn",
            "tạm biệt các bạn",
            # Kim Jong / random nonsense hallucinations
            "kim jong",
            "tỷ ngàn",
        ]

        all_segments = []
        full_texts = []
        MAX_CHUNK_SAMPLES = int(self.MAX_CHUNK_SEC * 16000)

        chunks_to_batch = []
        chunk_meta = [] # Store logic mapping

        for speech in speech_segs:
            seg_start = speech["start"]
            seg_end = speech["end"]

            s_idx = int(seg_start * 16000)
            e_idx = int(seg_end * 16000)
            seg_audio = audio[s_idx:e_idx]

            for j in range(0, len(seg_audio), MAX_CHUNK_SAMPLES):
                chunk = seg_audio[j: j + MAX_CHUNK_SAMPLES]
                chunk_start = seg_start + (j / 16000.0)
                chunk_end = chunk_start + (len(chunk) / 16000.0)
                chunk_dur = len(chunk) / 16000.0

                if chunk_dur < 0.2:
                    continue

                chunks_to_batch.append(chunk)
                chunk_meta.append({'start': chunk_start, 'end': chunk_end})

        if chunks_to_batch:
            batch_texts = self._transcribe_batch(chunks_to_batch, batch_size=6)
            
            for meta, text in zip(chunk_meta, batch_texts):
                if not text:
                    continue
                    
                txt_lower = text.lower()
                is_hallucination = False
                for h in HALLUCINATIONS:
                    if h in txt_lower:
                        is_hallucination = True
                        break

                if is_hallucination:
                    continue

                full_texts.append(text)
                all_segments.append({
                    "start": float(meta['start']),
                    "end": float(meta['end']),
                    "text": text
                })

        full_text = " ".join(full_texts).strip()

        if not all_segments and duration_sec < 5.0 and full_text:
            all_segments = [{"start": 0.0, "end": duration_sec, "text": full_text}]

        elapsed_ms = (time.time() - t_start) * 1000
        print(f"⏱️  Moonshine done: {elapsed_ms/1000:.2f}s | {len(all_segments)} segments")

        return {
            "text": full_text,
            "segments": all_segments,
            "total_ms": round(elapsed_ms, 1),
            "model": "Moonshine-base-vi",
        }


# ==========================================
# PYANNOTE DIARIZATION
# ==========================================

pyannote_pipeline = None
pyannote_lock = threading.Lock()  # Thêm Thread Lock để chống giẫm đạp CUDA
moonshine_lock = threading.Lock() # Lock riêng cho Moonshine STT để không chết chùm với Pyannote


def get_pyannote_pipeline():
    global pyannote_pipeline
    if pyannote_pipeline is None:
        print("🚀 Loading Pyannote 3.1 Pipeline...")
        from pyannote.audio import Pipeline

        base_path = os.path.dirname(os.path.abspath(__file__))

        # Dùng config.yaml đã có đầy đủ đường dẫn local (bao gồm plda)
        config_path = os.path.join(base_path, "models", "pyannote", "speaker-diarization-3.1", "config.yaml")

        pipeline = Pipeline.from_pretrained(config_path)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        if hasattr(pipeline, "segmentation_batch_size"):
            pipeline.segmentation_batch_size = 2  # Reduced drastically to free VRAM for Moonshine Live Fast models
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = 2  # Reduced to save VRAM

        pyannote_pipeline = pipeline
        print(f"✅ Pyannote loaded!")
    return pyannote_pipeline


async def run_diarize_first_pipeline(audio_source, stt_engine):
    """Full pipeline với Pyannote diarization (cho final/saved transcripts)."""
    start_time = time.time()

    # Load Audio
    if isinstance(audio_source, str) and os.path.exists(audio_source):
        audio, sr = load_audio_robust(audio_source)
    else:
        audio, sr = load_audio_robust(io.BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    duration = len(audio) / 16000.0

    # Enhance
    audio = enhance_audio_for_asr(audio)

    # Pyannote Diarization
    print("🔍 [Pipeline] Pyannote Diarization...")
    pipeline = get_pyannote_pipeline()
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    
    # Dùng Thread Lock ở đây để Cứu Pyannote khỏi Deadlock / VRAM OOM khi bị gọi bằng Multi-threading
    with pyannote_lock:
        print("🔒 [Lock] Khóa luồng CUDA chờ Pyannote Diarize...")
        diarization = pipeline({"waveform": waveform, "sample_rate": 16000})
        print("🔓 [Unlock] Đã nhả khỏa luồng CUDA.")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 🚀 Giải phóng bộ nhớ CUDA ngầm

    raw_spk_timeline = []
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_spk_timeline.append((turn.start, turn.end, speaker))

    if not raw_spk_timeline:
        print("⚠️ No speech detected for diarization.")
        return []

    # Merge segments per speaker
    spk_to_segs = {}
    for s, e, spk in raw_spk_timeline:
        if spk not in spk_to_segs:
            spk_to_segs[spk] = []
        spk_to_segs[spk].append((s, e))

    merged_timeline = []
    for spk, segs in spk_to_segs.items():
        segs.sort(key=lambda x: x[0])
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if s - cur_e <= 2.0 and (e - cur_s) < 25.0:
                cur_e = max(cur_e, e)
            else:
                if cur_e - cur_s >= 0.4:
                    merged_timeline.append((cur_s, cur_e, spk))
                cur_s, cur_e = s, e
        if cur_e - cur_s >= 0.4:
            merged_timeline.append((cur_s, cur_e, spk))

    merged_timeline.sort(key=lambda x: x[0])

    # Re-index labels
    label_map = {}
    next_id = 0
    renumbered_timeline = []
    for start, end, lbl in merged_timeline:
        if lbl not in label_map:
            label_map[lbl] = next_id
            next_id += 1
        renumbered_timeline.append((start, end, label_map[lbl]))
    merged_timeline = renumbered_timeline

    print(f"📊 Identified {len(label_map)} speakers in {len(merged_timeline)} segments")

    # Transcribe
    print("📝 [Pipeline] Transcribing segments...")
    final_output = []
    
    chunks_to_process = []
    chunk_meta = [] # mapping: { 'parent_idx': int, 'sub': bool }

    for i, (start, end, spk) in enumerate(merged_timeline):
        s_pad = 0.3
        if i > 0:
            prev_end = merged_timeline[i-1][1]
            if start - prev_end < 0.6:
                s_pad = max(0, (start - prev_end) / 2.0)

        e_pad = 0.3
        if i < len(merged_timeline) - 1:
            next_start = merged_timeline[i+1][0]
            if next_start - end < 0.6:
                e_pad = max(0, (next_start - end) / 2.0)

        s_idx = max(0, int((start - s_pad) * 16000))
        e_idx = min(len(audio), int((end + e_pad) * 16000))
        seg_audio = audio[s_idx:e_idx]

        duration_seg = len(seg_audio) / 16000.0
        
        # Placeholder cho final_output
        parent_idx = len(final_output)
        final_output.append({
            "start": float(start),
            "end": float(end),
            "speaker": f"Speaker {spk+1}",
            "text": ""
        })

        if duration_seg > 29.0:
            sub_chunks = stt_engine._get_speech_segments(seg_audio, duration_seg)
            for sub in sub_chunks:
                ss_idx = int(sub["start"] * 16000)
                ee_idx = int(sub["end"] * 16000)
                sub_audio = seg_audio[ss_idx:ee_idx]
                if len(sub_audio) / 16000.0 < 0.2:
                    continue
                chunks_to_process.append(sub_audio)
                chunk_meta.append({'parent_idx': parent_idx, 'sub': True})
        else:
            if duration_seg >= 0.2:
                chunks_to_process.append(seg_audio)
                chunk_meta.append({'parent_idx': parent_idx, 'sub': False})

    print(f"🚀 Batch inferencing {len(chunks_to_process)} chunks...")
    # BATCH PROCESS ALL CHUNKS VERY FAST
    # 🔥 Tăng Batch size lên 12 vì VRAM vừa dọn rác
    batch_texts = stt_engine._transcribe_batch(chunks_to_process, batch_size=12)
    
    # Map back to timeline
    for meta, text in zip(chunk_meta, batch_texts):
        if not text: continue
        p_idx = meta['parent_idx']
        if meta['sub']:
            if final_output[p_idx]["text"]:
                final_output[p_idx]["text"] += " " + text
            else:
                final_output[p_idx]["text"] = text
        else:
            final_output[p_idx]["text"] = text

    import re
    hallu_keywords = [
        # YouTube / subscribe hallucinations
        "ghiền mì", "youtube", "subscribe", "la la school",
        "đăng ký kênh", "để không bỏ lỡ", "những video hấp dẫn",
        "ủng hộ kênh của mình", "hãy đăng ký", "nhấn like",
        "đừng quên đăng ký", "chuông thông báo", "like và subscribe",
        # Common Vietnamese subtitle hallucinations
        "bạn đã xem video", "cảm ơn các bạn", "người dịch:",
        "subtitles by", "amara.org", "viết phụ đề bởi", "like và share",
        "hẹn gặp lại", "hẹn gặp lạ", "tạm biệt các bạn",
        # Random nonsense / noise hallucinations
        "kim jong", "tỷ ngàn",
    ]

    cleaned_output = []
    for item in final_output:
        text = item["text"]
        text = re.sub(r'(.{6,}?)(?:\s*\1){2,}', r'\1', text)
        t_lower = text.lower()
        is_hallucination = any(hk in t_lower for hk in hallu_keywords) or len(text.strip()) < 3
        
        if text and not is_hallucination:
            item["text"] = text.strip()
            cleaned_output.append(item)

    elapsed = time.time() - start_time
    print(f"✅ Full Pipeline Complete in {elapsed:.2f}s")
    return cleaned_output


# ==========================================
# SERVER STATE
# ==========================================

engine = MoonshineEngine()


@app.on_event("startup")
async def startup():
    try:
        engine.load()
    except Exception:
        print("⚠️ Failed to load model, will retry on request")



# ==========================================
# ASYNC TASK QUEUE ARCHITECTURE (SOLVING OOM & BLOCKING)
# ==========================================
import uuid
import threading
import queue
import asyncio

job_queue = queue.Queue() # Dành cho Full Diarization tasks (Dài, tốn VRAM)
live_job_queue = queue.Queue() # Dành riêng cho Live Streaming Chunks (Ngắn, Cần ưu tiên)
job_results = {}

def full_gpu_worker_thread():
    """
    Background worker for Full Meeting (diarization=true).
    """
    print("👷 [FULL VRAM] Worker thread started, waiting for jobs...")
    while True:
        job = job_queue.get()
        if job is None: break 
        
        task_id = job["task_id"]
        try:
            print(f"\\n[FULL-Worker] 🚀 Starting job {task_id}")
            job_results[task_id]['status'] = 'processing'
            
            audio_path = job.get("audio_path")
            audio_data = job.get("audio_data") # backward support
            response_format = job.get("response_format", "json")
            
            if not engine.loaded:
                engine.load()
                
            # Final/saved transcripts with Pyannote diarization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_source = audio_path if audio_path else audio_data
            segments = loop.run_until_complete(run_diarize_first_pipeline(audio_source, engine))
            loop.close()
            
            final_text = " ".join([seg["text"] for seg in segments if "text" in seg])
            final_res = {
                "text": final_text,
                "segments": segments,
                "model": "moonshine",
                "language": "vi"
            }
            
            job_results[task_id]['status'] = 'completed'
            job_results[task_id]['result'] = final_res
            print(f"[FULL-Worker] ✅ Finished job {task_id}")

        except Exception as e:
            print(f"[FULL-Worker] ❌ Error on job {task_id}: {e}")
            import traceback
            traceback.print_exc()
            job_results[task_id]['status'] = 'failed'
            job_results[task_id]['error'] = str(e)
            
        finally:
            if 'audio_path' in job and job['audio_path'] and os.path.exists(job['audio_path']):
                try: os.unlink(job['audio_path'])
                except: pass
            job_queue.task_done()

def live_gpu_worker_thread():
    """
    Background worker for Live Streaming (diarization=false).
    """
    print("👷 [LIVE VRAM] Worker thread started, waiting for jobs...")
    while True:
        job = live_job_queue.get()
        if job is None: break 
        
        task_id = job["task_id"]
        try:
            job_results[task_id]['status'] = 'processing'
            
            audio_path = job.get("audio_path")
            audio_data = job.get("audio_data")
            response_format = job.get("response_format", "json")
            
            if not engine.loaded:
                engine.load()
                
            # Live transcripts (chỉ VAD + transcribe)
            audio_source = audio_path if audio_path else audio_data
            result = engine.transcribe(audio_source)
            
            # Format response
            text = result.get('text', '')
            if response_format == "json":
                final_res = {"text": text}
            elif response_format == "text":
                final_res = text
            elif response_format == "verbose_json":
                final_res = {
                    "task": "transcribe",
                    "language": "vi",
                    "duration": result.get('total_ms', 0) / 1000.0,
                    "text": text,
                    "segments": result.get('segments', []),
                    "model": result.get("model", "moonshine")
                }
            else:
                final_res = {"text": text}

            job_results[task_id]['status'] = 'completed'
            job_results[task_id]['result'] = final_res
            print(f"[LIVE-Worker] ✅ Finished job {task_id}")
            
        except Exception as e:
            print(f"[LIVE-Worker] ❌ Error on job {task_id}: {e}")
            import traceback
            traceback.print_exc()
            job_results[task_id]['status'] = 'failed'
            job_results[task_id]['error'] = str(e)
            
        finally:
            if 'audio_path' in job and job['audio_path'] and os.path.exists(job['audio_path']):
                try: os.unlink(job['audio_path'])
                except: pass
            live_job_queue.task_done()

# Start Multiple GPU Workers (Khéo léo chia làn chống kẹt xe)
NUM_FULL_WORKERS = 2  # Dành cho tính năng "Kết thúc cuộc họp" (Nặng VRAM). Đặt = 1 để xếp hàng xử lý tuần tự chống crash/deadlock nội bộ Pyannote.
NUM_LIVE_WORKERS = 6  # Luôn để dành riêng 2 slots cho live-streaming (Nhẹ)

for i in range(NUM_FULL_WORKERS):
    worker = threading.Thread(target=full_gpu_worker_thread, daemon=True, name=f"Full-Worker-{i+1}")
    worker.start()

for i in range(NUM_LIVE_WORKERS):
    worker = threading.Thread(target=live_gpu_worker_thread, daemon=True, name=f"Live-Worker-{i+1}")
    worker.start()

print(f"🚀 Started {NUM_FULL_WORKERS} FULL Audio Workers and {NUM_LIVE_WORKERS} LIVE Chunk Workers!")

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    diarization: str = Form("false"),
    async_mode: str = Form("false"),
):
    """
    Standard Transcription Endpoint with internal queue waiting.
    Delegates work to GPU worker and waits for completion without blocking the event loop.
    """
    import tempfile
    import shutil
    
    # Kéo file trực tiếp xuống ổ cứng thay vì nhét RAM 1 cục để chống tràn quá tải
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    task_id = str(uuid.uuid4())
    
    # Store initial state
    job_results[task_id] = {
        "status": "queued",
        "result": None,
        "error": None
    }
    
    # Routing thông minh dựa trên flag diarization, truyền ĐƯỜNG DẪN file thay vì nguyên mảng Byte
    job_payload = {
        "task_id": task_id,
        "audio_path": temp_path,
        "response_format": response_format
    }
    
    if diarization.lower() == "true":
        job_queue.put(job_payload)  # Ném vào làn đường dành cho xe Container (Full Audio)
    else:
        live_job_queue.put(job_payload) # Ném vào làn đường dành cho xe Máy (Vài giây Live)
    
    if async_mode.lower() == "true":
        return {"task_id": task_id, "status": "processing"}
        
    # Wait for the background worker to finish this specific job
    import asyncio
    while True:
        if task_id in job_results:
            res = job_results[task_id]
            if res["status"] == "completed":
                result = job_results.pop(task_id)["result"]
                return result
            elif res["status"] == "failed":
                error_msg = job_results.pop(task_id)["error"]
                raise HTTPException(status_code=500, detail=f"Job failed: {error_msg}")
        await asyncio.sleep(0.5)

@app.get("/v1/audio/transcriptions/{task_id}")
async def get_transcribe_status(task_id: str):
    """
    Query the status of the background task and get the result if completed.
    """
    if task_id not in job_results:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return job_results[task_id]

@app.delete("/v1/audio/transcriptions/{task_id}")
async def delete_transcribe_task(task_id: str):
    """
    Clean up job results to free RAM.
    """
    if task_id in job_results:
        del job_results[task_id]
        return {"status": "success", "message": "Task cleaned up"}
    return {"status": "not_found"}

# Kẹp lại cái __main__ để test local
if __name__ == "__main__":
    import uvicorn
    # Use environment port or default to 8179
    port = int(os.environ.get("PORT", 8179))
    print(f"Starting server on port {port}...")
    uvicorn.run("service:app", host="0.0.0.0", port=port, reload=False)
