#!/bin/bash
# =============================================================================
# run_pipeline.sh — Chạy Golden Whisper + Diarization pipeline
# =============================================================================
# Sử dụng:
#   bash run_pipeline.sh audio_full.m4a
#   bash run_pipeline.sh audio_full.m4a --no-diarize
#   CONDA_ENV=asr bash run_pipeline.sh audio_full.m4a
# =============================================================================

set -e

AUDIO="${1:?Usage: bash run_pipeline.sh <audio_file> [--no-diarize]}"
EXTRA_ARGS="${@:2}"
CONDA_ENV="${CONDA_ENV:-asr}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kiểm tra audio file
if [ ! -f "${AUDIO}" ]; then
    echo "❌ Không tìm thấy file: ${AUDIO}"
    exit 1
fi

echo "========================================"
echo " ASR Pipeline — Golden Whisper + Diarization"
echo " Audio  : ${AUDIO}"
echo " Env    : conda:${CONDA_ENV}"
echo "========================================"

# Kiểm tra conda env
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "❌ Conda env '${CONDA_ENV}' chưa tồn tại."
    echo "   Chạy: bash setup_server.sh"
    exit 1
fi

# Chạy pipeline
conda run -n "${CONDA_ENV}" \
    python "${SCRIPT_DIR}/run_full_meeting.py" \
    --audio "${AUDIO}" \
    ${EXTRA_ARGS}
