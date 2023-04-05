#!/bin/bash

IMAGE_NAME=localhost/nn_ac_image

docker build -t $IMAGE_NAME -f docker/dockerfile.base ./ --build-arg path=$PWD