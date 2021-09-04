#FROM opencvcourses/opencv-docker:4.5.1
FROM python:3.9.7-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY lib/ lib/
COPY main.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD [ "python", "main.py", "help" ]
