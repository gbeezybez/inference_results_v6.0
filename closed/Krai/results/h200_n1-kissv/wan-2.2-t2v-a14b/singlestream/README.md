# MLPerf Inference v6.0 - Closed - Krai

To run experiments individually, use the following commands.

## h200_n1-kissv - wan-2.2-t2v-a14b - singlestream

### Accuracy  
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200,\
loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly,\
sage_offset_layers=4
```

### Performance 
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200,\
loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=57000,\
sage_offset_layers=4
```

### Compliance
#### TEST04
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200,\
loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=57000,\
loadgen_compliance_test=TEST04,loadgen_min_query_count=20,loadgen_mode_2=2,\
loadgen_performance_issue_unique=0,loadgen_performance_issue_same=1,loadgen_performance_issue_same_index=3,\
sage_offset_layers=4
```
