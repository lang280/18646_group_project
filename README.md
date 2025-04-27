


### CUDA version
```
nvcc -o new.x new.c forward.cu backward.cu -lm -arch=sm_75
```


### Run
```
./new.x
```

### Original version
```
gcc -o ori.x ori.c -lm
```

### Run original version
```
./ori.x
```

