FROM tensorflow/tensorflow:devel-gpu

## Copy resources to the Docker Image
COPY mrcnn/ /mnt/

## Install python3 pip
RUN apt-get install -y --no-install-recommends python3-pip

## Update pip
RUN pip3 install --upgrade pip

## Change the working Directory
WORKDIR /mnt/mrcnn

## Build package
RUN bash build.sh

## Install package 
RUN pip3 install bin/mrcnn-1.0.dev0.tar.gz

## Run the ship data praparation
RUN mrcnn train_ship train --dataset=./datasets --weights=last 