# MLPerf Inference v6.0 - Open - Krai

To run experiments individually, use the following commands.

## h200_n1-kissv - wan-2.2-t2v-a14b - offline

### Accuracy  
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,\
sage_offset_layers=4,caching_strategy=mag,division=open
```

### Performance 
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=0.025,\
sage_offset_layers=4,caching_strategy=mag,division=open
```
