#!/bin/bash

if (( "$(find /model -name Llama-3.1-8B* | wc -l)" > 0 )); then
  echo "AUTODETECTED: MODEL=llama3_1-8b"
  sleep 2
  bash calibration/quantize_8b.sh
elif (( "$(find /model -name Llama-2-70b* | wc -l)" > 0 )); then
  echo "AUTODETECTED: MODEL=llama2-70b"
  sleep 2
  bash calibration/quantize_70b.sh
else
  echo "ERROR: Model file not detected in /model. Exiting."
  exit
fi
