# 基于CUDA加速的深度神经网络（适用于Ubuntu系统）

该项目使用CUDA加速神经网络前向传播，可处理256×256尺寸的输入图像。

## 编译程序

### 依赖项
- CUDA Toolkit (>=10.0)
- GCC编译器

### 编译

```bash
# CUDA加速版本 (首选)
nvcc -o main main.c forward.cu -lm -arch=sm_70

# 如果目标GPU compute capability为其他版本，请修改sm_XX
# 例如，对于RTX 2080 Ti (compute capability 7.5)
# nvcc -o main main.c forward.cu -lm -arch=sm_75

# CPU-only版本 (不含CUDA加速)
# gcc -o main main.c -lm
```

## 运行程序

```bash
./main
```
