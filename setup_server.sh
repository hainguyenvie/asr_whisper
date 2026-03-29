#!/bin/bash
# =============================================================================
# setup_server.sh — Cài đặt môi trường trên server (không Docker)
# =============================================================================
# Sử dụng:
#   bash setup_server.sh [cuda_version]
#   bash setup_server.sh 121   # CUDA 12.1 (mặc định)
#   bash setup_server.sh 118   # CUDA 11.8
# =============================================================================

set -e

CUDA_VER="${1:-121}"
ENV_NAME="asr"
MODELS_DIR="$(dirname "$0")/models"

echo "========================================"
echo " ASR Server Setup"
echo " CUDA version: ${CUDA_VER}"
echo " Conda env   : ${ENV_NAME}"
echo "========================================"

# 1. Kiểm tra conda
if ! command -v conda &> /dev/null; then
    echo "❌ conda không tìm thấy. Hãy cài Miniconda/Anaconda trước."
    exit 1
fi

# 2. Tạo conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✅ Conda env '${ENV_NAME}' đã tồn tại, bỏ qua tạo mới."
else
    echo "📦 Tạo conda env '${ENV_NAME}' với Python 3.10..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

# 3. Activate và cài deps
echo "📥 Cài PyTorch (CUDA ${CUDA_VER})..."
if [ "${CUDA_VER}" = "121" ]; then
    conda run -n "${ENV_NAME}" pip install --no-cache-dir \
        torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
elif [ "${CUDA_VER}" = "118" ]; then
    conda run -n "${ENV_NAME}" pip install --no-cache-dir \
        torch==2.5.1+cu118 torchaudio==2.5.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  CUDA version không nhận ra: ${CUDA_VER}. Dùng CPU build..."
    conda run -n "${ENV_NAME}" pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1
fi

echo "📥 Cài faster-whisper..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir faster-whisper

echo "📥 Cài pyannote.audio (no-deps rồi fill deps)..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-deps pyannote.audio==4.0.4
conda run -n "${ENV_NAME}" pip install --no-cache-dir -r "$(dirname "$0")/requirements-standalone.txt"

# 4. Kiểm tra models
echo ""
echo "========================================"
echo " Kiểm tra models..."
echo "========================================"

check_model() {
    local path="$1"
    local name="$2"
    if [ -d "${path}" ] || [ -f "${path}" ]; then
        echo "  ✅ ${name}"
    else
        echo "  ❌ ${name} — THIẾU: ${path}"
    fi
}

check_model "${MODELS_DIR}/silero-vad"                                                      "Silero VAD"
check_model "${MODELS_DIR}/pyannote/speaker-diarization-3.1/config.yaml"                   "Pyannote config"
check_model "${MODELS_DIR}/pyannote/segmentation-3.0/pytorch_model.bin"                    "Pyannote segmentation"
check_model "${MODELS_DIR}/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"      "WeSpeaker embedding"

WHISPER_DIR="${HOME}/.cache/whisper-ct2/large-v3-turbo"
check_model "${WHISPER_DIR}"  "Whisper large-v3-turbo (CTranslate2)"

echo ""
echo "========================================"
echo " Xong! Chạy pipeline:"
echo "   bash run_pipeline.sh /path/to/audio.m4a"
echo "========================================"
