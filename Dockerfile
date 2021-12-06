FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.8-py3

RUN mkdir /detectron2
RUN apt-get update && apt-get install -y python3-opencv
RUN python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
WORKDIR /detectron2

COPY ./test.py .
COPY ./test_video.py .
COPY ./dummy.jpg .

RUN python3 test.py dummy.jpg dummy_out.jpg


