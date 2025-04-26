基于ubuntu22测试运行


### CUDA加速版本
```
nvcc -o main main.c forward.cu -lm -arch=sm_75
```
(sm_75 是 NVIDIA T4 GPU 的计算能力)
