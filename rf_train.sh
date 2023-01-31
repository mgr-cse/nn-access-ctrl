#!/bin/bash

dataset_folders="u4k-r4k-auth11k u4k-r4k-auth22k u4k-r6k-auth28k u4k-r7k-auth20k u5k-r5k-auth12k u5k-r5k-auth19k u6k-r6k-auth32k"
echo $dataset_folders to be trained!

cd ml_approach

for d in $dataset_folders; do
    # execute python script
    python rf_classify.py ../dataset/synthetic/${d}/train_${d}.sample ../dataset/synthetic/${d}/test_${d}.sample
    echo ++++ train complete!: $d
    # copy results folder
    mkdir -p ../02-output/rf/$d
    cp -r results ../02-output/rf/$d/
    echo ++++ results copied: $d

done