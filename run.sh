docker run --gpus all \
  -p 8000:8000 \
  -v "$(pwd)":/app \
  nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 \
  /bin/bash
