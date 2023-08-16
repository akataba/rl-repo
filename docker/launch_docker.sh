#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build -t rlrepo --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $(readlink -m $SCRIPT_DIR)
docker run --rm --gpus all --shm-size=16gb --network=host -v $(readlink -m $SCRIPT_DIR/../):/workdir/rl-repo -it rlrepo

