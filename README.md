


### CUDA version
```
nvcc -o new new.c forward.cu backward.cu -lm -arch=sm_75
```


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

