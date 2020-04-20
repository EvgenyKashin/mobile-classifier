#!/bin/bash
docker run --rm -it -v `pwd`:/workspace \
    mobile_classifier \
    python predict.py --folder_path $1
