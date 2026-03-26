git clone https://github.com/vllm-project/vllm -b v0.8.5.post1
cd vllm
pip install -r requirements/cpu.txt
VLLM_TARGET_DEVICE=cpu python setup.py install
cd .. && rm -rf vllm
