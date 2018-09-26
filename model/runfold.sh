#!/bin/bash

WHICH=nprf_drmm.py

for i in {1,2,3,4,5}
do
    CUDA_VISIBLE_DEVICES="" python $WHICH --fold $i temp1
done

