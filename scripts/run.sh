#!/bin/bash
export WANDB_PROJECT=GIT_Llama
export PROJECT_NAME=opt/exp001
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
