# Meetily AI Model Serving (Moonshine)

This directory contains the configuration to deploy the Moonshine Speech-to-Text service as a Docker container.

## Features
- **OpenAI Compatible API**: Drop-in replacement for OpenAI Whisper API (`/v1/audio/transcriptions`).
- **Moonshine ASR**: Vietnamese-optimized speech recognition model.
- **GPU/CPU Support**: CUDA-enabled for fast inference.
- **Silero VAD**: Voice Activity Detection for accurate speech segmentation.
- **Pyannote Diarization**: Speaker diarization for multi-speaker meetings.
- **Easy Deployment**: Standard `docker-compose` setup.

## Prerequisites
1. **Docker & Docker Compose** installed.
2. **NVIDIA GPU** recommended for optimal performance.
3. **Models**: You MUST have the models in the `models` folder.

   Structure:
   ```text
   serving/
   ├── models/
   │   ├── moonshine-base-vi/
   │   │   ├── model.safetensors
   │   │   ├── tokenizer.json
   │   │   ├── config.json
   │   │   └── preprocessor_config.json
   │   ├── silero-vad/
   │   │   ├── hubconf.py
   │   │   └── src/silero_vad/data/silero_vad.onnx
   │   ├── pyannote/
   │   │   ├── speaker-diarization-3.1/config.yaml
   │   │   ├── segmentation-3.0/pytorch_model.bin
   │   │   └── wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
   │   └── speaker/
   │       └── 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
   ├── docker-compose.yml
   └── Dockerfile
   ```

## Quick Start

1. Navigate to this directory:
   ```bash
   cd serving
   ```

2. Ensure your `models` folder is populated (see Prerequisites).

3. Build and start the service:
   ```bash
   docker-compose up -d --build
   ```

4. The service will be available at `http://localhost:2202`.

## API Endpoints

### POST /inference
Standard transcription with optional diarization.

**Parameters:**
- `file`: Audio file (required)
- `diarize`: Enable speaker diarization (default: "true")
- `temperature`: Sampling temperature (default: "0.0")
- `response_format`: Output format (default: "json")

**Example:**
```bash
curl http://localhost:2202/inference \
  -F "file=@meeting.wav" \
  -F "diarize=true"
```

### POST /v1/audio/transcriptions
OpenAI-compatible endpoint.

**Example:**
```bash
curl http://localhost:2202/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### POST /process_full_meeting
Full pipeline with Pyannote diarization-first approach.

**Example:**
```bash
curl http://localhost:2202/process_full_meeting \
  -F "file=@long_meeting.wav"
```

### GET /current_model
Returns current model in use.

**Example:**
```bash
curl http://localhost:2202/current_model
# Response: {"current_model": "moonshine"}
```

## Docker Compose

```yaml
version: '3.8'
services:
  speech-to-text:
    container_name: moonshine_stt_server
    build: .
    ports:
      - "2202:2202"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Environment Variables
- `PYTHONUNBUFFERED=1`: Enable real-time logging
- `HF_HUB_OFFLINE=1`: Force offline mode (no HuggingFace downloads)
- `TRANSFORMERS_OFFLINE=1`: Force offline mode for transformers

## Response Format

### Standard Transcription
```json
{
  "text": "Full transcription text",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "Segment text"}
  ],
  "total_ms": 5200,
  "device": "huggingface-cuda",
  "model": "Moonshine-base-vi"
}
```

### With Diarization
```json
{
  "text": "[SPEAKER_00]: Hello [SPEAKER_01]: Hi there",
  "segments": [
    {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 4.0, "text": "Hi there", "speaker": "SPEAKER_01"}
  ]
}
```