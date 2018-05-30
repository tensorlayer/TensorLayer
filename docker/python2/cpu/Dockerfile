FROM tensorflow/tensorflow:latest

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG TL_VERSION

RUN if [ -z "$TL_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get update \
        && apt-get install -y git \
        && mkdir /tensorlayer_dist/ \
        && cd /tensorlayer_dist/ \
        && git clone https://github.com/tensorlayer/tensorlayer.git \
        && cd tensorlayer \
        && pip install -e .[db,dev,doc,extra,test] \
        && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Building Tag Release:" "$TL_VERSION" \
        && pip install tensorlayer=="$TL_VERSION" ; \
    fi