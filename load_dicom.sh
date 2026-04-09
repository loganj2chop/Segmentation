#!/bin/bash


PYSCRIPT="train.py --fold $SPLIT_ID" 


conda run --no-capture-output -n MAG3 python $PYSCRIPT
