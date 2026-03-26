*Check [MLC MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.17.0-14-generic-x86_64-with-glibc2.39
* CPU version: x86_64
* Python version: 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=ebf22a2b1998b3b5371f02f70a076985219d8aa1


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: d986d93afc0b-nvidia-gpu-TensorRT-default_config

Model Precision: int8

### Accuracy Results 
`acc`: `76.078`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`Samples per second`: `87945.5`
