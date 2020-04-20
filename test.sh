#!/bin/bash
docker run --rm -it --gpus 1 -v `pwd`:/workspace \
    -v $1:/data --shm-size=2gb mobile_classifier \
    python test.py --checkpoint_path $2
