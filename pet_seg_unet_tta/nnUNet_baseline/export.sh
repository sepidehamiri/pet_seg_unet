#!/usr/bin/bash

sh build.sh

docker save autopet_baseline | gzip -c > autoPET_baseline.tar.gz
