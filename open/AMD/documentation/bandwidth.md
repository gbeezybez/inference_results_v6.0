# Bandwidth Requirement

Formulas for minimum network bandwidth are as [follows](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b1-ingress-bandwidth): 
 

Llama3.1-405b: $throughput * 20000 * dtype size$


Modern servers use PCIe gen5, which provides throughput of 4 gigabyte/second per lane and 64 GB/s for x16 connection. Assuming that
a NIC is connected via a PCIe x16 to the DRAM and data types used in our submission, the maximum supported throughput is well above what was found in our submission

| Benchmark |   Precision |       QPS    |
| --------- |-------------|--------------|
| Llama3.1-405b| 8 bit| 64 GB/(20000 * 1 Byte) = 3200000 |
 