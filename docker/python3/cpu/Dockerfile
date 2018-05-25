FROM tensorflow/tensorflow:latest-py3

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG TL_VERSION=false

RUN if [ "$TL_VERSION" = "false" ] ; then \
        echo 'Not Building with release'; \
    else \
        echo Installing Tensorlayer Version: "$TL_VERSION"; \
        pip install tensorlayer=="$TL_VERSION"; \
    fi