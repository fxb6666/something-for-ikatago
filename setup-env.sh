#!/bin/bash

KATAGO_BACKEND=${1^^}
WEIGHT_FILE=$2
RELEASE_VERSION=v1.4.1
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
[ -z "$GPU_NAME" ] && { printf '%b\n' '\e[33;1mWARN: Unknown GPU.\e[0m'; sleep 3; }

source /etc/os-release
if [ "$VERSION_ID" == "22.04" ] && [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" -O cuda-keyring.deb
elif [ "$VERSION_ID" == "24.04" ] && [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" -O cuda-keyring.deb
fi
if [ -f ./cuda-keyring.deb ]; then
  dpkg -i cuda-keyring.deb
  rm -f cuda-keyring.deb
fi
apt-get update -qq -y

mkdir -p /content
cd /content
wget -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/$RELEASE_VERSION/work.zip" -O work.zip
unzip -qq -o work.zip
cd work

# download binaries
if [[ ! $KATAGO_BACKEND =~ ^TENSORRT$|^CUDA$ ]]; then
  echo -e "\e[1;33m\nWARN: KATAGO_BACKEND=\"$KATAGO_BACKEND\" is invalid. Changed to \"CUDA\".\e[0m\n"
  KATAGO_BACKEND="CUDA"
fi
if [[ $KATAGO_BACKEND == TENSORRT ]]; then
  wget -T 10 -t 5 "https://github.com/lightvector/KataGo/releases/download/v1.18.1/katago-v1.18.1-trt10.2.0-cuda12.5-linux-x64.zip" -O ./katago.zip
elif [[ $KATAGO_BACKEND == CUDA ]]; then
  wget -T 10 -t 5 "https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.5-cudnn9.8.0-linux-x64.zip" -O ./katago.zip
fi
unzip -od data/bins ./katago.zip
chmod 755 ./data/bins/katago
cp ./data/bins/gtp_human5k_example.cfg ./data/configs/human_gtp.cfg

# download weights
wget -q -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/$RELEASE_VERSION/kata-weights.py" -O kata-weights.py
python3 ./kata-weights.py "$WEIGHT_FILE" "$KATAGO_BACKEND"
wget --retry-on-http-error=500 -T 10 -t 5 -c -O ./data/weights/human.bin.gz "https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz"

# install dependencies
ldconfig &>/dev/null
if [ "$KATAGO_BACKEND" == "CUDA" ]; then
  if ! ldconfig -p |grep 'libcudnn\.so\.9' &>/dev/null; then
    apt-get install -y --allow-change-held-packages --allow-downgrades libcudnn9-cuda-12=9.8.0.87-1
  fi
  if ! ldconfig -p |grep 'libcublas\.so\.12' &>/dev/null; then
    apt-get download -y libcublas-12-5
    dpkg -x libcublas-12-5*.deb .
    mv ./usr /
    rm -f libcublas-12-5*.deb
    echo '/usr/local/cuda-12.5/lib64' >/etc/ld.so.conf.d/libcublas12.conf
  fi
elif [ "$KATAGO_BACKEND" == "TENSORRT" ]; then
  if ! ldconfig -p |grep 'libnvinfer\.so\.10' &>/dev/null; then
    wget "https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/libnvinfer10_10.2.0.19-1+cuda12.5_amd64.deb" -O libnvinfer10.deb
    wget "https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/libnvonnxparsers10_10.2.0.19-1+cuda12.5_amd64.deb" -O libnvonnxparsers10.deb
    dpkg -i libnvinfer10.deb libnvonnxparsers10.deb
    rm -f libnvinfer10.deb libnvonnxparsers10.deb
  fi
  if [ "$GPU_NAME" == "TeslaT4" ] && env |grep "^COLAB" &>/dev/null; then
    wget -q -T 10 -t 5 "https://github.com/fxb6666/something-for-ikatago/releases/download/v1.4.0-file/trtcache.tar.xz" -O trtcache.tar.xz
    tar -C ~ -xf trtcache.tar.xz
  fi
fi
ldconfig &>/dev/null

chmod 755 ./ikatago-server
