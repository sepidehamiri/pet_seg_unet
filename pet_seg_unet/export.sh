#!/usr/bin/env bash

./build.sh

docker save pet_seg_unet | gzip -c > pet_seg_unet.tar.gz
