#!/bin/bash

FOLDER_NAME=$(basename "$PWD")
IMAGE_NAME="$FOLDER_NAME"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONTAINER_NAME="${FOLDER_NAME}_${TIMESTAMP}"

DOCKERFILE="Dockerfile"  # 기본 Dockerfile (prod)
# 운영체제 구분
UNAME=$(uname -s)

if [[ "$UNAME" == MINGW* || "$UNAME" == CYGWIN* || "$UNAME" == MSYS* ]]; then
    # Windows (Git Bash 등)
    HOST_DIR=$(pwd -W)  # Windows 절대경로로 변환
    echo "[*] Windows 환경 감지됨 → HOST_DIR=${HOST_DIR}"
else
    # Linux / WSL / macOS
    HOST_DIR=$PWD
    echo "[*] Linux/macOS 환경 감지됨 → HOST_DIR=${HOST_DIR}"
fi

# Docker 이미지가 없으면 빌드 (모드별 Dockerfile 사용)
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "[+] Building Docker image '$IMAGE_NAME' with $DOCKERFILE ..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

echo "[*] 개발모드: 전체 디렉토리 마운트"
docker run -it --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$HOST_DIR":/app \
    "$IMAGE_NAME"
