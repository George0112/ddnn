FROM python:3.7.10-slim
#FROM tensorflow/tensorflow:2.1.0
WORKDIR /dnn

RUN apt-get update && apt-get upgrade && apt-get install -y gfortran \
  libhdf5-dev libc-ares-dev libeigen3-dev \
  libatlas-base-dev libopenblas-dev libblas-dev \
  liblapack-dev 

RUN pip install pybind11 Cython==0.29.21

RUN pip install h5py==2.10.0

RUN pip install --upgrade setuptools

RUN pip install gdown

RUN cp ~/.local/bin/gdown /usr/local/bin/gdown

RUN gdown https://drive.google.com/uc?id=1fR9lsi_bsI_npPFB-wZyvgjbO0V9FbMf

RUN pip install tensorflow-2.2.0-cp37-cp37m-linux_aarch64.whl

COPY requirement.txt /dnn/

#RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl

RUN pip install -r requirement.txt

COPY . ./

CMD ["python", "run.py", "vgg16", "0", "0"]
