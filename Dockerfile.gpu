# https://store.docker.com/community/images/tensorflow/tensorflow
FROM tensorflow/tensorflow:latest-gpu

ADD . /tensorlayer
RUN pip install /tensorlayer
ENV PYTHONUNBUFFERED=1
