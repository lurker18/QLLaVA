#!/bin/bash
export WANDB_PROJECT=GIT_Llama
export PROJECT_NAME=llama/exp001
export WANDB_NAME=$PROJECT_NAME

python train.py --config_file projects/$PROJECT_NAME.yml