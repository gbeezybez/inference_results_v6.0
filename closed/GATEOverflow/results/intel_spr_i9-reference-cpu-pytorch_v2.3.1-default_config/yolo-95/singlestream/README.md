*Check [MLC MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-94-generic-x86_64-with-glibc2.39
* CPU version: x86_64
* Python version: 3.12.3 (main, Jan 22 2026, 20:57:42) [GCC 13.3.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo GATEOverflow@mlperf-automations --checkout=9758d8ff6ef9bc779dc7aa14f73701c1bb14d2a2


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload GATEOverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo GATEOverflow@mlperf-automations
mlc pull repo GATEOverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: intel_spr_i9-reference-cpu-pytorch_v2.3.1-default_config

Model Precision: fp32

### Accuracy Results 
`mAP`: `53.108`, Required accuracy for closed division `>= 50.73`

### Performance Results 
`90th percentile latency (ns)`: `90105803.0`
