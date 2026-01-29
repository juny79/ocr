FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# 의존성 설치
RUN apt-get update && apt-get install -y python3-pip git

# 프로젝트 복사
COPY . .

# 패키지 설치
RUN pip install -r baseline_code/requirements.txt

# W&B API 키 설정 (build-time)
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV WANDB_PROJECT=ocr-receipt-detection

# 학습 실행
CMD ["python", "baseline_code/runners/train.py", "preset=example", "wandb=True"]
