


### CUDA version
```
nvcc -o new new.c forward.cu backward.cu -lm -arch=sm_75
```
(sm_75 是 NVIDIA T4 GPU 的计算能力)

### Run
```
./new
```

### Original version
```
gcc -o ori ori.c -lm
```

### Run original version
```
./ori
```

