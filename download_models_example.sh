#!/bin/bash

# Create models directory structure
mkdir -p models/moonshine-base-vi
mkdir -p models/silero-vad
mkdir -p models/pyannote/speaker-diarization-3.1
mkdir -p models/pyannote/segmentation-3.0
mkdir -p models/pyannote/wespeaker-voxceleb-resnet34-LM
mkdir -p models/speaker

# Download Moonshine-base-vi (Vietnamese)
# Option 1: From HuggingFace Hub (requires HF_TOKEN if private)
# huggingface-cli download UsefSensors/moonshine-base-vi --local-dir models/moonshine-base-vi

# Option 2: If you have local copy, copy from there
# cp -r /path/to/moonshine-base-vi/* models/moonshine-base-vi/

# Download Silero VAD
git clone https://github.com/snakers4/silero-vad.git models/silero-vad

# Download Pyannote models (requires HF_TOKEN)
# pip install pyannote.audio
# python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"

# Download Speaker Recognition (3D-Speaker)
# wget -P models/speaker/ https://huggingface.co/csukuangfj/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k/resolve/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo "=========================================="
echo "Models should be placed in:"
echo "  - models/moonshine-base-vi/ (Moonshine ASR)"
echo "  - models/silero-vad/ (VAD)"
echo "  - models/pyannote/ (Diarization)"
echo "  - models/speaker/ (Speaker Recognition)"
echo "=========================================="
echo ""
echo "Required files for Moonshine-base-vi:"
echo "  - model.safetensors"
echo "  - tokenizer.json"
echo "  - config.json"
echo "  - preprocessor_config.json"
echo "  - generation_config.json"