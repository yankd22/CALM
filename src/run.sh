#!/bin/bash

commands=(
  "CUDA_VISIBLE_DEVICES=0 python main.py --name_list '[[1,2,3,4,5,6,7],[8]]'"
  "CUDA_VISIBLE_DEVICES=1 python main.py --name_list '[[1,2,3,4,5,6,8],[7]]'"
)

# 使用 xargs 每次并行执行两个命令
printf "%s\n" "${commands[@]}" | xargs -I CMD -n 1 -P 1 bash -c CMD