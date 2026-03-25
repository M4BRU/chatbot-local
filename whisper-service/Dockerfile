FROM python:3.12-slim

WORKDIR /app

# ffmpeg requis par faster-whisper pour le décodage audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge le modèle au build (évite download au runtime)
ARG WHISPER_MODEL=large-v3-turbo
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY app.py .
COPY blocklist_fr.txt .

ENV WHISPER_MODEL=large-v3-turbo
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
ENV WHISPER_LANGUAGE=fr

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
