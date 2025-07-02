import subprocess
import os
import time

def run_nnunet():
    # 현재 디렉토리 이름으로부터 도커 컨테이너 이름 생성
    image_name = os.path.basename(os.getcwd())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    container_name = f"{image_name}_{timestamp}"


    # 3. Docker run 명령어 구성
    docker_command = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--name", container_name,
        "-v", f"{os.getcwd()}:/app",  # 현재 디렉토리 마운트
        image_name,
        "bash", "-c", "chmod +x /app/nnunet_run.sh && /app/nnunet_run.sh"
    ]

    result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)


run_nnunet()

