xhost +  && docker run --gpus all --env="QT_X11_NO_MITSHM=1" --env NVIDIA_DISABLE_REQUIRE=1 -it --rm --privileged --network=host --name cleargrasp  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/wy/cleargrasp/:/home/cleargrasp/ -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --ipc=host -e DISPLAY=${DISPLAY}  -p 8000:8001 yong123/cleargrasp

docker run --gpus all -it --rm \\
    --privileged \\
    --user=$(id -u $USER):$(id -g $USER) \\
    --env="DISPLAY" \\
    --workdir="/home/$USER" \\
    --volume="/home/$USER:/home/$USER" \\
    --volume="/etc/group:/etc/group:ro" \\
    --volume="/etc/passwd:/etc/passwd:ro" \\
    --volume="/etc/shadow:/etc/shadow:ro" \\
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \\
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \\
    -e DISPLAY=$DISPLAY --name glvnd cuda_gui

xhost + && docker run --name cleargrasp -p 8000:8001 -v /home/ubuntu/Users/yong/.clear/:/home/cleargrasp/ --network host --ipc host -e DISPLAY=${DISPLAY} -e NVIDIA_DISABLE_REQUIRE=1 -e QT_X11_NO_MITSHM=1 --cap-add SYS_PTRACE --security-opt seccomp=unconfined --gpus all -it --rm -v /tmp:/tmp yong123/cleargrasp

# 查看端口占用情况：sudo netstat -anp |grep 8000
