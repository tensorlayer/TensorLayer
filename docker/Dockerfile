# Build args.
#   * Accepted Values:
#        - Python 2 + CPU: "latest"          =>  --build-arg TF_CONTAINER_VERSION="latest"
#        - Python 2 + GPU: "latest-gpu"      =>  --build-arg TF_CONTAINER_VERSION="latest-gpu"
#        - Python 3 + CPU: "latest-py3"      =>  --build-arg TF_CONTAINER_VERSION="latest-py3"
#        - Python 3 + GPU: "latest-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-gpu-py3"
ARG TF_CONTAINER_VERSION="latest"

FROM tensorflow/tensorflow:${TF_CONTAINER_VERSION}

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG TL_VERSION

RUN if [ -z "$TL_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get update \
        && apt-get install -y git python3-tk python-tk \
        && if [ python -c "import sys; print(sys.version_info >= (3,))" = 'True' ]; then apt-get install -y python3-tk; else apt-get install -y python-tk; fi \
        && mkdir /tensorlayer_dist/ \
        && cd /tensorlayer_dist/ \
        && git clone https://github.com/tensorlayer/tensorlayer.git \
        && cd tensorlayer \
        && pip install --upgrade -e .[all] \
        && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Building Tag Release:" "$TL_VERSION" \
        && apt-get update \
        && if [ python -c "import sys; print(sys.version_info >= (3,))" = 'True' ]; then apt-get install -y python3-tk; else apt-get install -y python-tk; fi \
        && pip install --upgrade tensorlayer[all]=="$TL_VERSION" ; \
    fi
