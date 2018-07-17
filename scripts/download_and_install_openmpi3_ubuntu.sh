#!/bin/sh

set -e

if [ $(uname) == "Darwin" ]; then
	NPROC=$(sysctl -n hw.ncpu)
else
	NPROC=$(nproc)
fi

mkdir -p $HOME/openmpi_tmp && cd $HOME/openmpi_tmp

# TODO: upgrade to latest version once https://github.com/open-mpi/ompi/pull/5296 is in the release
MPI_MAJOR=3
MPI_MINOR=1

VERSION=${MPI_MAJOR}.${MPI_MINOR}.0
FILENAME=openmpi-${VERSION}.tar.bz2
FOLDER=openmpi-${VERSION}
URL=https://download.open-mpi.org/release/open-mpi/v${MPI_MAJOR}.${MPI_MINOR}/${FILENAME}

[ ! -f ${FILENAME} ] && curl -vLOJ $URL
tar -xf ${FILENAME}
cd ${FOLDER}

# will take about 8 min or longer depends on your machine
./configure --prefix=$HOME/local/openmpi
make -j ${NPROC} all
make install

rm -rf $HOME/openmpi_tmp

echo 'Update the PATH with OpenMPI bin by running: PATH=$PATH:$HOME/local/openmpi/bin'
echo 'Update the PATH in ~/.bashrc if you want OpenMPI to be ready once the machine start'
