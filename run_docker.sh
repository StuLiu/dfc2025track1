

sudo su

docker pull registry.ap-northeast-1.aliyuncs.com/liuwang20144623/dfc2025track1:v1

docker images

docker run -it --shm-size=60g --gpus all [image_id] /bin/bash

cd /workspace/DFC2025Track1

bash run_report.sh
