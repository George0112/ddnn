#FROM python:3.7.10-slim

FROM tensorflow/tensorflow:2.1.0

WORKDIR /dnn

COPY requirement.txt /dnn/

RUN pip install -r requirement.txt

COPY . ./

CMD ["python", "run.py", "vgg16", "0", "0", "--is-last"]
