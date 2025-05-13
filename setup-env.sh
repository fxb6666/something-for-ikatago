#!/bin/bash

KATAGO_BACKEND=${1^^}
WEIGHT_FILE=$2
USE_HIGHTHREADS=$3
RELEASE_VERSION=v1.2.0
GPU_NAME=`nvidia-smi -q | grep "Product Name" | cut -d":" -f2 | tr -cd '[:alnum:]._-'`
#GPU_NAME=TeslaT4

if [ "$KATAGO_BACKEND" == "AUTO" ]; then
  KATAGO_BACKEND="CUDA"
fi
if [ "$KATAGO_BACKEND" == "TRT" ]; then
  KATAGO_BACKEND="TENSORRT"
fi
echo "Using GPU: " $GPU_NAME
echo "Using Katago Backend: " $KATAGO_BACKEND
echo "Using Katago Weight: " $WEIGHT_FILE

mkdir -p /content
cd /content
apt-get update -qq -y
apt-get install -qq -y libzip-dev
if [[ -f /usr/lib/x86_64-linux-gnu/libzip.so.4 ]] && [[ ! -f /usr/lib/x86_64-linux-gnu/libzip.so.5 ]]; then
  ln -sf /usr/lib/x86_64-linux-gnu/libzip.so.4 /usr/lib/x86_64-linux-gnu/libzip.so.5
fi
wget --quiet -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/$RELEASE_VERSION/work.zip" -O work.zip
unzip -qq -o work.zip
cd work

#download the binarires
if [[ ! $KATAGO_BACKEND =~ ^TENSORRT$|^CUDA$ ]]; then
  echo -e "\e[1;33m\nWARN: KATAGO_BACKEND=\"$KATAGO_BACKEND\" is invalid. Changed to \"CUDA\".\e[0m\n"
  KATAGO_BACKEND="CUDA"
fi
if [[ $KATAGO_BACKEND == TENSORRT ]]; then
  wget -q -T 10 -t 5 "https://github.com/lightvector/KataGo/releases/download/v1.16.0/katago-v1.16.0-trt10.2.0-cuda12.5-linux-x64.zip" -O ./katago.zip
elif [[ $KATAGO_BACKEND == CUDA ]]; then
  wget -q -T 10 -t 5 "https://github.com/lightvector/KataGo/releases/download/v1.16.0/katago-v1.16.0-cuda12.5-cudnn8.9.7-linux-x64.zip" -O ./katago.zip
fi
unzip -od data/bins ./katago.zip
chmod +x ./data/bins/katago
cp ./data/bins/gtp_human5k_example.cfg ./data/configs/human_gtp.cfg

#download the weights
wget -nv -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/$RELEASE_VERSION/kata-weights.py" -O kata-weights.py
python3 ./kata-weights.py "$WEIGHT_FILE" "$KATAGO_BACKEND"
wget -T 10 -t 5 -c -O ./data/weights/human.bin.gz "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"

if ! ldconfig -p |grep 'libcudnn\.so\.8' &>/dev/null; then
  apt-get install -y libcudnn8=8.9.7.29-1+cuda12.2
fi
if ! ldconfig -p |grep 'libcublas\.so\.12' &>/dev/null; then
  apt-get download -y libcublas-12-5
  dpkg -x libcublas-12-5*.deb /
  rm -f libcublas-12-5*.deb
  echo '/usr/local/cuda-12.5/lib64' >/etc/ld.so.conf.d/libcublas12.conf
fi
if [ "$KATAGO_BACKEND" == "TENSORRT" ]; then
  apt-get install -y libnvinfer10=10.2.0.19-1+cuda12.5
  if [ "$GPU_NAME" == "TeslaT4" ] && env |grep "^COLAB" &>/dev/null; then
    wget -q -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/v1.0.0/timing-caches.tar.xz" -O timing-caches.tar.xz
    mkdir -p ~/.katago/trtcache
    tar -C ~/.katago/trtcache/ -xf timing-caches.tar.xz
  fi
fi
if ! ldconfig -p |grep 'libssl\.so\.1\.1' &>/dev/null; then
  wget -T 10 -t 5 -qO libssl1.1.deb "https://github.com/fxb6666/something-for-ikatago/releases/download/v1.1.2-libssl1.1/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb"
  dpkg -i libssl1.1.deb
fi
ldconfig

chmod +x ./ikatago-server
