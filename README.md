# Seek for Commonalities: Shared Features Extraction for Multi-task Reinforcement Learning via Adversarial Training

This repository contains a reference implementation for Adaptive Experience buffer and Shared Feature extractor Multi-Task Reinforcement Learning (AESF-MTRL). 

## Requirements

1.ray

2.mujoco_py

3.gym_dmcontrol

4.dmcontrol

5.gym

6.torch

7.redis_py (you should also install the redis)

## Training

Before training, you should set the corresponding parameter in `main.py` 

`python SFAE/main.py` 

## Evaluation

change the `isTrain` to false