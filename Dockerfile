FROM tensorflow/tensorflow:2.2.0

WORKDIR /dnn

COPY . ./

RUN pip install -r requirement.txt

CMD ["python3", "-u", "run.py"]
