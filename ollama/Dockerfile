FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ollama v0.17.7
RUN curl -fsSL https://github.com/ollama/ollama/releases/download/v0.17.7/ollama-linux-amd64.tar.zst \
    -o /tmp/ollama.tar.zst \
    && cd /tmp \
    && tar -I zstd -xf ollama.tar.zst \
    && install -m 755 bin/ollama /usr/local/bin/ollama \
    && rm -rf /tmp/ollama.tar.zst /tmp/bin /tmp/lib

EXPOSE 11434

ENV OLLAMA_HOST=0.0.0.0

CMD ["ollama", "serve"]
