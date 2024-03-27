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
apt install --yes libzip4 1>/dev/null
rm -rf work
wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/work.zip -O work.zip
unzip -qq work.zip

cd /content/work
mkdir -p /content/work/data/bins
mkdir -p /content/work/data/weights

#download the binarires
if [[ ! $KATAGO_BACKEND =~ TENSORRT|CUDA ]]; then
  echo -e "\e[1;33m\nWARN: KATAGO_BACKEND=\"$KATAGO_BACKEND\" is invalid. Changed to \"CUDA\".\e[0m\n"
  KATAGO_BACKEND="CUDA"
fi
if [[ $KATAGO_BACKEND == TENSORRT ]]; then
  wget -nv "https://github.com/lightvector/KataGo/releases/download/v1.14.1/katago-v1.14.1-trt8.6.1-cuda12.1-linux-x64.zip" -O ./katago.zip
elif [[ $KATAGO_BACKEND == CUDA ]]; then
  wget -nv "https://github.com/lightvector/KataGo/releases/download/v1.14.1/katago-v1.14.1-cuda12.1-cudnn8.9.7-linux-x64.zip" -O ./katago.zip
fi
unzip -od data/bins ./katago.zip
chmod +x ./data/bins/katago
mkdir -p /root/.katago/
cp -r ./opencltuning /root/.katago/

#download the weights
wget -nv "https://raw.githubusercontent.com/fxb6666/something-for-ikatago/main/kata-weights.py" -O kata-weights.py
python3 ./kata-weights.py "$WEIGHT_FILE" "$KATAGO_BACKEND"

cp /root/.katago/opencltuning/tune6_gpuTeslaK80_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaK80_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c384_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c384_mv10.txt
cp /root/.katago/opencltuning/tune8_gpuTeslaK80_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune8_gpuTeslaK80_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune8_gpuTeslaP100PCIE16GB_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune8_gpuTeslaP100PCIE16GB_x19_y19_c256_mv10.txt

if [ "$KATAGO_BACKEND" == "TENSORRT" ]
then
  #  apt-get install libnvinfer8=8.2.0-1+cuda11.4
  wget -q "https://github.com/fxb6666/something-for-ikatago/releases/download/v1.0.0/timing-caches.zip" -O timing-caches.zip
  unzip -qod / timing-caches.zip
  apt-get install libnvinfer8=8.6.1.6-1+cuda12.0
fi

chmod +x ./ikatago-server
