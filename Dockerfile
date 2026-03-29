FROM python:3.10-slim

# ===== Copy & trust internal CA =====
COPY swg_ca.crt /usr/local/share/ca-certificates/swg_ca.crt
RUN chmod 644 /usr/local/share/ca-certificates/swg_ca.crt \
    && update-ca-certificates

# ===== Proxy & cert env =====
ENV http_proxy="http://10.208.187.1:3128/"
ENV https_proxy="http://10.208.187.1:3128/"
ENV HTTP_PROXY="http://10.208.187.1:3128/"
ENV HTTPS_PROXY="http://10.208.187.1:3128/"
ENV FTP_PROXY="http://10.208.187.1:3128/"

ENV NO_PROXY="127.*,10.*,192.168.*,localhost,10.207.163.17,10.254.139.*,10.240.171.230,10.240.144.197,10.240.173.77,10.254.139.24,10.254.139.25"
ENV no_proxy="$NO_PROXY"

ENV REQUESTS_CA_BUNDLE="/usr/local/share/ca-certificates/swg_ca.crt"
ENV SSL_CERT_FILE="/usr/local/share/ca-certificates/swg_ca.crt"
ENV CURL_CA_BUNDLE=""

# ===== Offline mode (no HuggingFace downloads) =====
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# ===== pip config =====
RUN pip config set global.cert "/usr/local/share/ca-certificates/swg_ca.crt" \
    && pip config set global.trusted-host "pypi.python.org pypi.org files.pythonhosted.org" \
    && pip install --upgrade pip \
    && pip install requests==2.27.1 \
    && pip install --upgrade requests

# ===== System tools =====
RUN apt-get update \
    && apt-get install -y nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== Audio deps =====
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ===== Python deps =====
COPY requirements-serving.txt .
# Install torch first with CUDA 12.1 build (compatible with host driver 535 / CUDA 12.2)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
# Install pyannote.audio 4.x (requires PLDA/VBx support) without pulling torch>=2.8.0
# PLDA is pure NumPy/SciPy so works fine with torch 2.5.1; audio passed as waveform tensor so torchcodec not needed
RUN pip install --no-cache-dir --no-deps pyannote.audio==4.0.4
RUN pip install --no-cache-dir -r requirements-serving.txt

# ===== App code =====
COPY service.py .

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "9120", "--timeout-keep-alive", "1200"]
