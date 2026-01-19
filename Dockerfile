# Topology-Efficient Deep Learning 실험 이미지
#
# Build:
#   docker build -t topology-dl-experiments .
#
# Run locally:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
#     topology-dl-experiments python experiments/track_a/train.py --model ph_mlp --dataset ECG200

FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ src/
COPY experiments/ experiments/
COPY configs/ configs/
COPY scripts/ scripts/

# 데이터 및 결과 디렉토리
RUN mkdir -p data results

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 기본 명령어
CMD ["python", "experiments/track_a/train.py", "--help"]
