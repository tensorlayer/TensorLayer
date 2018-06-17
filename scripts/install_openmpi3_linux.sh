set -e

if [ $(uname) == "Darwin" ]; then
	NPROC=$(sysctl -n hw.ncpu)
else
	NPROC=$(nproc)
fi

mkdir -p ./openmpi_tmp && cd ./openmpi_tmp

MPI_MAJOR=3
MPI_MINOR=1

VERSION=${MPI_MAJOR}.${MPI_MINOR}.0
FILENAME=openmpi-${VERSION}.tar.bz2
FOLDER=openmpi-${VERSION}
URL=https://download.open-mpi.org/release/open-mpi/v${MPI_MAJOR}.${MPI_MINOR}/${FILENAME}

[ ! -f ${FILENAME} ] && curl -vLOJ $URL
tar -xf ${FILENAME}
cd ${FOLDER}

# real	5m7.636s on 64 core machine
./configure --prefix=$HOME/tensorlayerlib/openmpi
make -j ${NPROC} all
make install

echo 'Update the PATH with OpenMPI bin by running: PATH=$PATH:$HOME/tensorlayerlib/openmpi/bin'
echo 'Update the PATH in ~/.bashrc if you want OpenMPI to be ready once the machine start'
echo 'You can safely delete the ./openmpi_tmp folder now.'