# This script is for installing horovod on readthedocs only!
set -e

pwd
SCRIPT_DIR=$(cd $(dirname $0) && pwd)

[ ! -z "$1" ] && export PATH=$1:$PATH

LOCATION=/home/docs
URL=https://github.com/lgarithm/openmpi-release/raw/master/releases/openmpi-bin-3.1.0-rtd.tar.bz2

mkdir -p ${LOCATION}
chmod a+rx ${LOCATION}
cd ${LOCATION}
curl -vLOJ ${URL}
tar -xf *.tar.bz2

pip install tensorflow==1.5.0 # must install tensorflow before horovod
PATH=${LOCATION}/openmpi/bin:$PATH pip install horovod

# install all requirements except tensorflow
for req in $(find $SCRIPT_DIR/../requirements -type f); do
    if [ ! $(grep tensorflow $req) ]; then
        pip install -r $req
    fi
done

echo "done $0"
