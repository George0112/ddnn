FROM tensorflow/tensorflow:2.1.0

WORKDIR /dnn

COPY . ./

RUN pip install -r requirement.txt

CMD ["python", "run.py", "vgg16", "0", "0"]
