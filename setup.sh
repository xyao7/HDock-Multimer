# Bash script for installing the tools required by HDM

# Install FFT library
if ! ldconfig -p | grep -q "libfftw3.so.3"; then
    apt-get install -y -qq libfftw3-double3
fi

# Install HDOCKlite
wget -q http://huanglab.phys.hust.edu.cn/software/hdocklite/download/145978343768fa3d6897522/HDOCKlite.tar.gz
tar -xzf HDOCKlite.tar.gz
cp HDOCKlite-v1.1/createpl tools/
cp HDOCKlite-v1.1/hdock tools/
rm -rf HDOCKlite.tar.gz HDOCKlite-v1.1/

# Install HSYMDOCKlite
wget -q http://huanglab.phys.hust.edu.cn/software/hsymdock/download/154078528868fa3d63dc36a/HSYMDOCKlite.tar.gz
tar -xzf HSYMDOCKlite.tar.gz
cp HSYMDOCK_v1.1/chdock tools/
cp HSYMDOCK_v1.1/compcn tools/
cp HSYMDOCK_v1.1/dhdock tools/
cp HSYMDOCK_v1.1/dhdock.sh tools/
cp HSYMDOCK_v1.1/compdn tools/
rm -rf HSYMDOCKlite.tar.gz HSYMDOCK_v1.1/

# Install jq
wget -q https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux64
chmod +x jq-linux64
mv jq-linux64 tools/jq

# Install MMalign
wget -q https://github.com/pylelab/USalign/archive/refs/heads/master.zip -O USalign-master.zip
unzip -q USalign-master.zip
cd USalign-master/
g++ -O3 -ffast-math -lm -o MMalign MMalign.cpp
mv MMalign ../tools/
cd ../ && rm -rf USalign-master.zip USalign-master/

# Install STRIDE
wget -q https://github.com/heiniglab/stride/archive/refs/heads/main.zip -O stride-main.zip
unzip -q stride-main.zip
mv stride-main/stride tools/
rm -rf stride-main.zip stride-main/

