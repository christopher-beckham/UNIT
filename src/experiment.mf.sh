#!/bin/bash

source activate pytorch-env-py36
python cocogan_train.py --config ../exps/unit/male2female.yaml --log ../results --resume=1
