
## 7/17

* Basic tensor implementation
* CPU device
* Metal device on Mac
* Windows build
* CUDA reference implementation

## 7/18

* Tensor operation
* Cutlass + Cublas
* Benchmark (w/o any optimization)

```
CPU
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.012        0.012        0.600        1.377          3.375
sub            128      50        0.015        0.015        0.751        1.092         -0.125
matmul         128      50        0.664        0.680       33.985        6.320          8.594
add            256      50        0.048        0.048        2.420        1.368         -0.750
sub            256      50        0.059        0.061        3.058        1.101         -0.500
matmul         256      50        9.248        9.748      487.385        3.628         -7.891
add            512      50        0.190        0.202       10.115        1.380         -1.500
sub            512      50        0.238        0.242       12.102        1.100         -2.000
matmul         512      50      114.319      115.626     5781.283        2.348        151.250
add           1024      50        0.759        0.829       41.460        1.382          1.000
sub           1024      50        0.960        0.997       49.874        1.093          0.500
matmul        1024      50     2393.043     2462.873   123143.654        0.897         66.406

CUDA (reference)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.008        0.017        0.838        2.008          3.375
sub            128      50        0.008        0.015        0.758        2.024         -0.125
matmul         128      50        0.026        0.051        2.572      164.457          8.594
add            256      50        0.010        0.017        0.873        6.671         -0.750
sub            256      50        0.008        0.016        0.781        8.127         -0.500
matmul         256      50        0.026        0.042        2.103     1286.596         -7.891
add            512      50        0.010        0.016        0.798       27.036         -1.500
sub            512      50        0.010        0.018        0.903       27.582         -2.000
matmul         512      50        0.106        0.116        5.815     2532.792        151.250
add           1024      50        0.012        0.019        0.964       85.556          1.000
sub           1024      50        0.011        0.026        1.303       94.705          0.500
matmul        1024      50        0.728        0.763       38.152     2949.192         66.406

CUDA (cuBLAS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.007        0.013        0.640        2.265          3.375
sub            128      50        0.025        0.043        2.163        0.656         -0.125
matmul         128      50        0.016        0.034        1.716      261.100          8.594
add            256      50        0.008        0.015        0.731        8.063         -0.750
sub            256      50        0.009        0.014        0.690        7.699         -0.500
matmul         256      50        0.032        0.053        2.670     1049.626         -7.891
add            512      50        0.026        0.044        2.193       10.266         -1.500
sub            512      50        0.010        0.018        0.903       25.924         -2.000
matmul         512      50        0.033        0.060        3.013     8120.628        151.250
add           1024      50        0.012        0.016        0.822       88.802          1.000
sub           1024      50        0.023        0.037        1.866       45.322          0.500
matmul        1024      50        0.092        0.094        4.701    23334.097         66.406

CUDA (CUTLASS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.008        0.020        1.018        1.992          3.375
sub            128      50        0.008        0.015        0.737        1.977         -0.125
matmul         128      50        0.018        0.020        1.013      227.951          8.594
add            256      50        0.009        0.018        0.906        7.642         -0.750
sub            256      50        0.009        0.015        0.755        7.699         -0.500
matmul         256      50        0.026        0.028        1.414     1274.090         -7.891
add            512      50        0.010        0.018        0.880       27.398         -1.500
sub            512      50        0.009        0.018        0.904       30.341         -2.000
matmul         512      50        0.045        0.047        2.326     5996.146        151.250
add           1024      50        0.014        0.026        1.302       74.304          1.000
sub           1024      50        0.012        0.019        0.945       87.850          0.500
matmul        1024      50        0.080        0.081        4.071    26843.546         66.406
```

* Optimize CPU with multi-threads

```
CPU (reference)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.012        0.012        0.605        1.365          3.375
sub            128      50        0.016        0.016        0.806        1.044         -0.125
matmul         128      50        0.655        0.681       34.071        6.407          8.594
add            256      50        0.047        0.052        2.578        1.383         -0.750
sub            256      50        0.062        0.064        3.186        1.052         -0.500
matmul         256      50        9.343        9.703      485.155        3.591         -7.891
add            512      50        0.190        0.209       10.466        1.381         -1.500
sub            512      50        0.252        0.260       13.022        1.039         -2.000
matmul         512      50      114.393      115.013     5750.669        2.347        151.250
add           1024       5        0.765        0.775        3.877        1.370          1.000
sub           1024       5        1.002        1.010        5.050        1.047          0.500
matmul        1024       5     3170.056     3183.299    15916.495        0.677         66.406

CPU (multi-thread, 24 threads)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.002        0.002        0.076       10.923          3.375
sub            128      50        0.002        0.002        0.076       10.923         -0.125
matmul         128      50        0.671        0.762       38.081        6.249          8.594
add            256      50        0.004        0.004        0.207       15.984         -0.750
sub            256      50        0.004        0.004        0.216       15.241         -0.500
matmul         256      50        1.237        1.381       69.073       27.123         -7.891
add            512      50        0.095        0.143        7.154        2.762         -1.500
sub            512      50        0.096        0.130        6.480        2.745         -2.000
matmul         512      50        7.946        8.481      424.060       33.782        151.250
add           1024       5        0.526        0.573        2.864        1.993          1.000
sub           1024       5        0.502        0.549        2.745        2.090          0.500
matmul        1024       5      152.703      157.299      786.497       14.063         66.406
```

* Optimize add/sub kernel

```
CUDA (reference)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.008        0.014        0.676        2.008          3.375
sub            128      50        0.009        0.016        0.812        1.829         -0.125
add            256      50        0.008        0.023        1.164        7.938         -0.750
sub            256      50        0.009        0.017        0.828        7.087         -0.500
add            512      50        0.011        0.020        0.992       23.207         -1.500
sub            512      50        0.009        0.017        0.871       30.007         -2.000
add           1024      50        0.012        0.030        1.508       88.802          1.000
sub           1024      50        0.024        0.044        2.180       43.344          0.500

CUDA (cuBLAS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.007        0.015        0.767        2.327          3.375
sub            128      50        0.025        0.040        1.980        0.666         -0.125
add            256      50        0.008        0.015        0.755        7.969         -0.750
sub            256      50        0.008        0.013        0.673        8.127         -0.500
add            512      50        0.008        0.015        0.726       31.629         -1.500
sub            512      50        0.008        0.015        0.736       31.752         -2.000
add           1024      50        0.012        0.018        0.882       87.850          1.000
sub           1024      50        0.011        0.016        0.820       97.815          0.500

CUDA (CUTLASS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.008        0.014        0.701        2.098          3.375
sub            128      50        0.008        0.015        0.733        2.142         -0.125
add            256      50        0.008        0.014        0.697        8.393         -0.750
sub            256      50        0.009        0.015        0.757        7.699         -0.500
add            512      50        0.008        0.016        0.783       31.387         -1.500
sub            512      50        0.010        0.030        1.522       25.680         -2.000
add           1024      50        0.011        0.017        0.838       95.256          1.000
sub           1024      50        0.010        0.017        0.849      101.764          0.500
```

* Added different stragies

```
CUDA elementwise kernel benchmark
Device: NVIDIA GeForce RTX 5070 Ti (70 SMs)
Tensor: 1024 x 1024 (1048576 Float32 elements)
Timing: 7 samples x 1000 launches; allocations, copies, and synchronization excluded
Bandwidth counts two reads plus one write (12 bytes/element).

Op      Configuration            Blocks      Threads   Elem/thr     Best us      Avg us         GB/s
add     exact grid                 4096      1048576       1.00        8.42        9.04      1494.08
add     grid-stride 1xSM             70        17920      58.51       10.40       10.64      1210.39
add     grid-stride 2xSM            140        35840      29.26        8.20        8.93      1534.63
add     grid-stride 4xSM            280        71680      14.63       11.37       12.07      1106.79
add     grid-stride 8xSM            560       143360       7.31        7.77        8.12      1620.26
add     grid-stride 16xSM          1120       286720       3.66        7.89        8.23      1595.12
add     grid-stride 32xSM          2240       573440       1.83        7.55        8.05      1665.58
add     fixed 2 elem/thread        2048       524288       2.00        7.83        8.20      1606.64
add     fixed 4 elem/thread        1024       262144       4.00        7.69        8.15      1635.83
add     fixed 8 elem/thread         512       131072       8.00        7.15        7.81      1760.59
add     fixed 16 elem/thread        256        65536      16.00        7.03        7.64      1790.73
add     fixed 32 elem/thread        128        32768      32.00       12.39       12.61      1015.51
sub     exact grid                 4096      1048576       1.00        8.97        9.53      1402.89
sub     grid-stride 1xSM             70        17920      58.51       10.40       10.75      1209.41
sub     grid-stride 2xSM            140        35840      29.26        8.49        9.04      1482.78
sub     grid-stride 4xSM            280        71680      14.63        8.01        8.50      1570.53
sub     grid-stride 8xSM            560       143360       7.31        7.22        7.98      1741.77
sub     grid-stride 16xSM          1120       286720       3.66        8.28       11.38      1519.79
sub     grid-stride 32xSM          2240       573440       1.83        7.37        7.80      1706.63
sub     fixed 2 elem/thread        2048       524288       2.00        7.51        8.06      1676.12
sub     fixed 4 elem/thread        1024       262144       4.00        7.52        7.92      1673.28
sub     fixed 8 elem/thread         512       131072       8.00       11.19       11.54      1124.73
sub     fixed 16 elem/thread        256        65536      16.00        7.33        7.80      1716.76
sub     fixed 32 elem/thread        128        32768      32.00        8.79        9.15      1431.34

CUDA elementwise kernel benchmark
Device: NVIDIA GeForce RTX 5070 Ti (70 SMs)
Tensor: 2048 x 2048 (4194304 Float32 elements)
Timing: 7 samples x 1000 launches; allocations, copies, and synchronization excluded
Bandwidth counts two reads plus one write (12 bytes/element).

Op      Configuration            Blocks      Threads   Elem/thr     Best us      Avg us         GB/s
add     exact grid                16384      4194304       1.00       32.70       33.00      1539.26
add     grid-stride 1xSM             70        17920     234.06       67.52       67.72       745.48
add     grid-stride 2xSM            140        35840     117.03       38.30       38.60      1314.08
add     grid-stride 4xSM            280        71680      58.51       24.65       24.84      2042.25
add     grid-stride 8xSM            560       143360      29.26       24.68       24.76      2039.78
add     grid-stride 16xSM          1120       286720      14.63       22.74       23.03      2213.74
add     grid-stride 32xSM          2240       573440       7.31       22.90       23.36      2197.70
add     fixed 2 elem/thread        8192      2097152       2.00       24.50       24.84      2054.27
add     fixed 4 elem/thread        4096      1048576       4.00       22.88       24.60      2199.53
add     fixed 8 elem/thread        2048       524288       8.00       22.85       23.30      2202.83
add     fixed 16 elem/thread       1024       262144      16.00       22.90       23.42      2198.09
add     fixed 32 elem/thread        512       131072      32.00       26.46       26.85      1902.33
sub     exact grid                16384      4194304       1.00       32.74       33.01      1537.33
sub     grid-stride 1xSM             70        17920     234.06       67.52       68.08       745.46
sub     grid-stride 2xSM            140        35840     117.03       38.28       38.70      1314.90
sub     grid-stride 4xSM            280        71680      58.51       24.65       24.74      2041.81
sub     grid-stride 8xSM            560       143360      29.26       24.69       24.74      2038.69
sub     grid-stride 16xSM          1120       286720      14.63       22.70       22.95      2216.97
sub     grid-stride 32xSM          2240       573440       7.31       23.25       23.35      2164.59
sub     fixed 2 elem/thread        8192      2097152       2.00       24.60       24.67      2046.31
sub     fixed 4 elem/thread        4096      1048576       4.00       22.78       23.28      2209.03
sub     fixed 8 elem/thread        2048       524288       8.00       23.02       23.98      2186.62
sub     fixed 16 elem/thread       1024       262144      16.00       23.60       24.01      2133.10
sub     fixed 32 elem/thread        512       131072      32.00       26.70       27.10      1885.03
```

* matmul kernel benchmark

```
CUDA square Float32 matmul kernel benchmark
Device: NVIDIA GeForce RTX 5070 Ti (70 SMs)
Matrices: 1024 x 1024
Timing: 5 samples x 20 launches; allocations, copies, and synchronization excluded

Configuration                 Blocks   Thr/block      Best ms       Avg ms      TFLOP/s
naive 8x8                      16384          64        0.864        0.869        2.487
naive 16x16 (current)           4096         256        0.717        0.733        2.995
naive 32x8                      4096         256        0.693        0.697        3.097
naive 8x32                      4096         256        0.885        0.889        2.427
naive 32x16                     2048         512        0.717        0.723        2.996
naive 16x32                     2048         512        0.779        0.783        2.755
tiled 8x8                      16384          64        0.689        0.693        3.119
tiled 16x16                     4096         256        0.533        0.537        4.029
tiled 32x32                     1024        1024        0.569        0.569        3.773

CUDA square Float32 matmul kernel benchmark
Device: NVIDIA GeForce RTX 5070 Ti (70 SMs)
Matrices: 2048 x 2048
Timing: 5 samples x 20 launches; allocations, copies, and synchronization excluded

Configuration                 Blocks   Thr/block      Best ms       Avg ms      TFLOP/s
naive 8x8                      65536          64        6.776        6.795        2.535
naive 16x16 (current)          16384         256        5.617        5.634        3.059
naive 32x8                     16384         256        5.457        5.473        3.148
naive 8x32                     16384         256        6.966        6.971        2.466
naive 32x16                     8192         512        5.578        5.587        3.080
naive 16x32                     8192         512        6.020        6.034        2.854
tiled 8x8                      65536          64        5.641        5.649        3.046
tiled 16x16                    16384         256        4.180        4.185        4.110
tiled 32x32                     4096        1024        4.466        4.484        3.847
```

* optimize matmul kernel using tiles

```
CUDA (reference)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.008        0.017        0.831        2.090          3.375
sub            128      50        0.008        0.015        0.772        2.098         -0.125
matmul         128      50        0.010        0.016        0.799      407.056          8.594
add            256      50        0.009        0.016        0.803        7.613         -0.750
sub            256      50        0.008        0.016        0.786        7.969         -0.500
matmul         256      50        0.018        0.021        1.026     1836.385         -7.891
add            512      50        0.010        0.017        0.835       27.036         -1.500
sub            512      50        0.009        0.015        0.747       30.682         -2.000
matmul         512      50        0.078        0.080        3.995     3447.846        151.250
add           1024      50        0.011        0.020        0.989       96.661          1.000
sub           1024      50        0.014        0.020        1.022       74.984          0.500
matmul        1024      50        0.535        0.570       28.497     4013.208         66.406

CUDA (cuBLAS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.010        0.017        0.865        1.668          3.375
sub            128      50        0.007        0.014        0.700        2.236         -0.125
matmul         128      50        0.016        0.024        1.181      258.016          8.594
add            256      50        0.008        0.014        0.688        8.192         -0.750
sub            256      50        0.009        0.017        0.871        7.670         -0.500
matmul         256      50        0.017        0.021        1.037     1967.310         -7.891
add            512      50        0.011        0.019        0.969       23.745         -1.500
sub            512      50        0.008        0.015        0.734       32.000         -2.000
matmul         512      50        0.031        0.037        1.850     8630.255        151.250
add           1024      50        0.011        0.018        0.882       98.997          1.000
sub           1024      50        0.011        0.018        0.914       93.623          0.500
matmul        1024      50        0.092        0.099        4.957    23285.518         66.406

CUDA (CUTLASS)
Operation        N    Runs      Best ms       Avg ms     Total ms      GFLOP/s       Checksum
add            128      50        0.009        0.018        0.877        1.855          3.375
sub            128      50        0.009        0.015        0.751        1.842         -0.125
matmul         128      50        0.019        0.020        1.015      224.055          8.594
add            256      50        0.008        0.021        1.046        7.728         -0.750
sub            256      50        0.008        0.016        0.794        8.095         -0.500
matmul         256      50        0.028        0.029        1.474     1187.515         -7.891
add            512      50        0.010        0.021        1.037       25.600         -1.500
sub            512      50        0.009        0.023        1.162       30.454         -2.000
matmul         512      50        0.045        0.046        2.310     5970.540        151.250
add           1024      50        0.012        0.016        0.804       90.519          1.000
sub           1024      50        0.010        0.017        0.865      100.515          0.500
matmul        1024      50        0.081        0.083        4.136    26535.732         66.406
```