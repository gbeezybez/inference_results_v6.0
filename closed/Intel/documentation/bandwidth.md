# Bandwidth specs of datacenter systems for MLPerf Inference 6.0

As per the [MLPerf Datacenter Bandwidth Requirements](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#appendix-b-datacenter-bandwidth-requirements) , Datacenter systems must satisfy both the ingress and egress bandwidth requirements for each benchmark.

Intel submitted using the following datacenter systems for respective workload submission:

| System                 | Workloads Submitted                            | Ethernet Controller               | Network Bandwidth |
|------------------------|------------------------------------------------|-----------------------------------|-------------------|
| Intel Xeon 6980P       | Llama3.1-8B, Whisper, RGAT                     | Intel Ethernet Controller I210    | 1.0 GbE           |
| Intel Arc Pro B50      | Llama3.1-8B, Whisper                           | Intel Ethernet Controller I225-LM | 2.5 GbE           |
| Intel Arc Pro B60      | Llama3.1-8B, Whisper, Llama2-70B, GPT-OSS-120B | Intel Ethernet Controller I226-LM | 2.5 GbE           |
| Intel Arc Pro B60 Dual | Llama3.1-8B, Whisper, Llama2-70B, GPT-OSS-120B | Intel Ethernet Controller I226-LM | 2.5 GbE           |
| Intel Arc Pro B70      | Llama3.1-8B, Whisper, Llama2-70B, GPT-OSS-120B | Intel Ethernet Controller I226-LM | 2.5 GbE           |


### Ingress bandwidth requirements

| Benchmark    | Formula                                                        | Bandwidth used   (bytes)     | Values                                                                 |
|--------------|----------------------------------------------------------------|------------------------------|------------------------------------------------------------------------|
| Llama3.1-8B  | ```used_bw = tput x max_input_len x dtype_size```              | ```used_bw = tput x 10240``` | ```max_input_len = 2560; dtype_size = 4B```                            |
| Llama2-70B   | ```used_bw = tput x max_input_len x dtype_size```              | ```used_bw = tput x 4096```  | ```max_input_len = 1024; dtype_size = 4B```                            |
| GPT-OSS-120B | ```used_bw = tput x max_input_len x dtype_size```              | ```used_bw = tput x 61320``` | ```max_input_len = 15330; dtype_size = 4B```                           |
| Whisper      | ```used_bw = tput x audio_length x sample_rate x dtype_size``` | ```used_bw = tput x 960000```| ```max_audio_length = 30sec; sample_rate = 16000Hz; dtype_size = 2B``` |
| R-GAT        | negligible                                                     | negligible                   | >0                                                                     |


## Max permissible QPS per system

Below table shows max permissible sample throughput per workload, calculated using the system bandwidth. Througput numbers below are well above the submission throughput.
Note: Llama3.1-8B, Llama2-70B, GPT-OSS-120B, and Whisper report their respective benchmark throughputs in tokens/sec, not QPS.


| System                                | Bandwidth of system (bytes/sec)| Llama3.1-8B | Llama2-70B | GPT-OSS-120B | Whisper | RGAT       |
|---------------------------------------|--------------------------------|-------------|------------|--------------|---------|------------|
| Intel Xeon 6980P                      | 1.250× 10^8                    | 12,207      | NA         | NA           | 130     | negligible |
| Intel Arc Pro B50, B60, B60 Dual, B70 | 3.125x 10^8                    | 30,517      | 76,293     | 5,096        | 325     | negligible |
