*Check [MLC MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.17.9-76061709-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo mlcommons@mlperf-automations --checkout=27d53d88511a984c88cc5a42ad6334ddc7c0546c


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload mlcommons@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo mlcommons@mlperf-automations
mlc pull repo mlcommons@mlperf-automations
mlc rm cache -f

```

## Results

Platform: Lenovo_IdeaPad_Gaming_3_Ryzen_5_4600H-reference-gpu-pytorch_v2.5.0a0-cu126

Model Precision: fp32

### Accuracy Results 
`mAP`: `53.108`, Required accuracy for closed division `>= 52.866`

### Performance Results 
`Samples per second`: `24.187`
