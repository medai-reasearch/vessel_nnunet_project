# 베이스 이미지 지정
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel


# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# # 현재 디렉토리의 모든 파일을 컨테이너로 복사
COPY ./nnUNet /app/nnUNet

# FastAPI와 Uvicorn 설치 + nnUNet 소스 설치
RUN pip install fastapi uvicorn[standard] \
    && pip install -e /app/nnUNet/.

# 쉘 환경으로 진입
CMD ["/bin/bash"]