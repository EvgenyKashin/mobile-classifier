FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# for opencv
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

RUN conda install -c conda-forge jupyter pandas matplotlib scikit-learn opencv
RUN pip install pytorch-lightning onnx onnxruntime timm facenet-pytorch

# install apex for fp16
RUN git clone https://github.com/NVIDIA/apex /apex && cd /apex && \
    pip install -v --no-cache-dir ./