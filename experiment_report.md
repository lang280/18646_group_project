Experiment with unrolled kernel
```
===== BENCHMARKING KERNEL TYPE 0 =====

Benchmarking original kernel with batch size 32...
Results for original kernel with batch size 32:
  Average time: 0.1139 ms
  Min time:     0.1125 ms
  Max time:     0.1161 ms
  Std dev:      0.0008 ms
  Performance metrics:
    Throughput:        280953.31 images/sec
    Computational:     56.39 GFLOPS
    Memory bandwidth:  9.11 GB/s

Benchmarking original kernel with batch size 64...
Results for original kernel with batch size 64:
  Average time: 0.1154 ms
  Min time:     0.1130 ms
  Max time:     0.1180 ms
  Std dev:      0.0008 ms
  Performance metrics:
    Throughput:        554503.44 images/sec
    Computational:     111.29 GFLOPS
    Memory bandwidth:  11.01 GB/s

Benchmarking original kernel with batch size 128...
Results for original kernel with batch size 128:
  Average time: 0.2178 ms
  Min time:     0.2160 ms
  Max time:     0.2304 ms
  Std dev:      0.0014 ms
  Performance metrics:
    Throughput:        587650.25 images/sec
    Computational:     117.94 GFLOPS
    Memory bandwidth:  7.98 GB/s

Benchmarking original kernel with batch size 256...
Results for original kernel with batch size 256:
  Average time: 0.4281 ms
  Min time:     0.4265 ms
  Max time:     0.4296 ms
  Std dev:      0.0006 ms
  Performance metrics:
    Throughput:        598014.56 images/sec
    Computational:     120.02 GFLOPS
    Memory bandwidth:  6.24 GB/s

Benchmarking original kernel with batch size 512...
Results for original kernel with batch size 512:
  Average time: 0.7360 ms
  Min time:     0.5250 ms
  Max time:     0.7536 ms
  Std dev:      0.0543 ms
  Performance metrics:
    Throughput:        695650.12 images/sec
    Computational:     139.62 GFLOPS
    Memory bandwidth:  6.17 GB/s

Benchmarking original kernel with batch size 1024...
Results for original kernel with batch size 1024:
  Average time: 0.9886 ms
  Min time:     0.8845 ms
  Max time:     1.0460 ms
  Std dev:      0.0629 ms
  Performance metrics:
    Throughput:        1035832.50 images/sec
    Computational:     207.90 GFLOPS
    Memory bandwidth:  8.37 GB/s

===== BENCHMARKING KERNEL TYPE 1 =====

Benchmarking loop-unrolled kernel with batch size 32...
Results for loop-unrolled kernel with batch size 32:
  Average time: 0.0883 ms
  Min time:     0.0868 ms
  Max time:     0.0900 ms
  Std dev:      0.0005 ms
  Performance metrics:
    Throughput:        362602.50 images/sec
    Computational:     72.78 GFLOPS
    Memory bandwidth:  11.75 GB/s

Benchmarking loop-unrolled kernel with batch size 64...
Results for loop-unrolled kernel with batch size 64:
  Average time: 0.0873 ms
  Min time:     0.0811 ms
  Max time:     0.0924 ms
  Std dev:      0.0035 ms
  Performance metrics:
    Throughput:        733487.19 images/sec
    Computational:     147.21 GFLOPS
    Memory bandwidth:  14.56 GB/s

Benchmarking loop-unrolled kernel with batch size 128...
Results for loop-unrolled kernel with batch size 128:
  Average time: 0.1471 ms
  Min time:     0.1425 ms
  Max time:     0.1504 ms
  Std dev:      0.0027 ms
  Performance metrics:
    Throughput:        870098.38 images/sec
    Computational:     174.63 GFLOPS
    Memory bandwidth:  11.81 GB/s

Benchmarking loop-unrolled kernel with batch size 256...
Results for loop-unrolled kernel with batch size 256:
  Average time: 0.2767 ms
  Min time:     0.2708 ms
  Max time:     0.2841 ms
  Std dev:      0.0029 ms
  Performance metrics:
    Throughput:        925268.62 images/sec
    Computational:     185.71 GFLOPS
    Memory bandwidth:  9.66 GB/s

Benchmarking loop-unrolled kernel with batch size 512...
Results for loop-unrolled kernel with batch size 512:
  Average time: 0.4926 ms
  Min time:     0.4753 ms
  Max time:     0.5325 ms
  Std dev:      0.0180 ms
  Performance metrics:
    Throughput:        1039481.44 images/sec
    Computational:     208.63 GFLOPS
    Memory bandwidth:  9.22 GB/s

Benchmarking loop-unrolled kernel with batch size 1024...
Results for loop-unrolled kernel with batch size 1024:
  Average time: 0.9887 ms
  Min time:     0.9114 ms
  Max time:     1.0363 ms
  Std dev:      0.0502 ms
  Performance metrics:
    Throughput:        1035712.56 images/sec
    Computational:     207.87 GFLOPS
    Memory bandwidth:  8.37 GB/s
```