FROM tensorflow/tensorflow:latest-gpu

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG TL_VERSION=false

RUN if [ "$argname" = "false" ] ; then
        echo 'Not Building with release'
    else
        pip install tensorlayer=="$TL_VERSION"
    fi