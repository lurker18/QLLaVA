#!/bin/bash
export WANDB_PROJECT=GIT_MPT
export PROJECT_NAME=mpt/exp001
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
