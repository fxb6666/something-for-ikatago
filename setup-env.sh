#!/bin/bash

KATAGO_BACKEND=$1
WEIGHT_FILE=$2
USE_HIGHTHREADS=$3
RELEASE_VERSION=2.0.0
GPU_NAME=`nvidia-smi -q | grep "Product Name" | cut -d":" -f2 | tr -cd '[:alnum:]._-'`
#GPU_NAME=TeslaT4

detect_auto_backend () {
  if [ "$GPU_NAME" == "TeslaT4" ]
  then
    KATAGO_BACKEND="CUDA"
  else
    KATAGO_BACKEND="OPENCL"
  fi
}

if [ "$KATAGO_BACKEND" == "AUTO" ]
then
  detect_auto_backend
fi

if [ "$KATAGO_BACKEND" == "TRT" ]
then
  KATAGO_BACKEND="TENSORRT"
fi
echo "Using GPU: " $GPU_NAME
echo "Using Katago Backend: " $KATAGO_BACKEND
echo "Using Katago Weight: " $WEIGHT_FILE

cd /content
apt-get install -qq -y libzip4
rm -rf work
wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/work.zip -O work.zip
unzip -qq work.zip

cd /content/work
mkdir -p /content/work/data/bins
mkdir -p /content/work/data/weights

#download the binarires
if [[ ! $KATAGO_BACKEND =~ ^TENSORRT$|^CUDA$ ]]; then
  echo -e "\e[1;33m\nWARN: KATAGO_BACKEND=\"$KATAGO_BACKEND\" is invalid. Changed to \"CUDA\".\e[0m\n"
  KATAGO_BACKEND="CUDA"
fi
if [[ $KATAGO_BACKEND == TENSORRT ]]; then
  wget -q "https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-trt10.2.0-cuda12.5-linux-x64.zip" -O ./katago.zip
elif [[ $KATAGO_BACKEND == CUDA ]]; then
  wget -q "https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda12.5-cudnn8.9.7-linux-x64.zip" -O ./katago.zip
fi
unzip -od data/bins ./katago.zip
chmod +x ./data/bins/katago
cp ./data/bins/gtp_human5k_example.cfg ./data/configs/human_gtp.cfg
mkdir -p /root/.katago/

#download the weights
wget -nv "https://raw.githubusercontent.com/fxb6666/something-for-ikatago/main/kata-weights.py" -O kata-weights.py
python3 ./kata-weights.py "$WEIGHT_FILE" "$KATAGO_BACKEND"
wget -O ./data/weights/human.bin.gz "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"

if [ "$KATAGO_BACKEND" == "TENSORRT" ]
then
  apt-get install -y libnvinfer10=10.2.0.19-1+cuda12.5
  if [ "$GPU_NAME" == "TeslaT4" ]; then
    wget -q "https://github.com/fxb6666/something-for-ikatago/releases/download/v1.0.0/timing-caches.zip" -O timing-caches.zip
    mkdir -p ~/.katago/trtcache
    unzip -qojd ~/.katago/trtcache timing-caches.zip
  fi
fi

ln -sf /usr/lib/x86_64-linux-gnu/libzip.so.4 /usr/lib/x86_64-linux-gnu/libzip.so.5
url=$(wget --retry-on-http-error=500 --timeout=6 -qO- "https://packages.ubuntu.com/focal/amd64/libssl1.1/download" | grep -o -E -m1 'http[^"]+amd64\.deb')
wget -nv -O libssl1.1.deb "$url"
dpkg -i libssl1.1.deb

chmod +x ./ikatago-server
