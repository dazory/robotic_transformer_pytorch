# DOCKER_NAME="jayhi/rt:rt-1-pytorch"
# DOCKER_NAME="dshong/rt:rt-1-pytorch"
DOCKER_NAME="dshong/rt:rt-1-pytorch-tf"

docker run -ti --rm --gpus all \
        -e CUDA_VISIBLE_DEVICES=$1 \
        --shm-size 32g \
        -v `pwd`:/ws/external \
        -v /media/TrainDataset:/ws/data \
        $DOCKER_NAME bash
