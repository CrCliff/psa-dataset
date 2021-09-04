FROM http://027517924056.dkr.ecr.us-east-1.amazonaws.com/python:3.9.7-slim

ENV S3_IN ""
ENV S3_OUT ""

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ecs/ ecs/
COPY lib/ lib/
COPY main.py .

CMD python3 main.py pr --fin="${S3_IN}" --fout="${S3_OUT}"
