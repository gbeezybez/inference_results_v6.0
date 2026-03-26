import os
import sys
import glob
import hashlib
from pathlib import Path

data = {
    "llama3_1-405b": {
        "model_path" : ["/model/llama3.1-405b/fp4_quantized", "/model/llama3.1-405b/fp8_quantized", "/model/llama3.1-405b/fp4_quantized/pruned_59_84"],
        "safetensor_md5sum" : ["7dccd6047bbf888c31bce62f588b9dbb", "9cf165d1e925964ff9936902d6db3613", "fd1d753bf4e8a98ce6cc7e39a9d49295"],
        "dataset_path" : "/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl",
        "dataset_md5sum" : "25d9f2085bbcacb07554cd7f4af303c9",
    },
    "llama3_1-405b-interactive": {
        "model_path" : ["/model/llama3.1-405b/fp4_quantized", "/model/llama3.1-405b/fp8_quantized", "/model/llama3.1-405b/fp4_quantized/pruned_59_84"],
        "safetensor_md5sum" : ["7dccd6047bbf888c31bce62f588b9dbb", "9cf165d1e925964ff9936902d6db3613", "fd1d753bf4e8a98ce6cc7e39a9d49295"],
        "dataset_path" : "/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl",
        "dataset_md5sum" : "25d9f2085bbcacb07554cd7f4af303c9",
    },
}


class ResourceChecker:


    def check(self, configs):
        result = []
        benchmark_name = configs["benchmark_name"]
        scenario = configs["scenario"]
        benchmark = data[benchmark_name]
        if isinstance(benchmark.get(scenario), dict):
            benchmark = benchmark.get(scenario)
        output_log_dir = configs["harness_config"]["output_log_dir"]

        invalid_file_path = output_log_dir + "/INVALID"
        if os.path.exists(invalid_file_path):
            os.remove(invalid_file_path)

        #validate model
        current_model_path = configs["llm_config"].get("model", configs["llm_config"].get("model_path", None))
        expected_model_paths = benchmark["model_path"]
        expected_model_md5sums = benchmark["safetensor_md5sum"]
        if current_model_path not in expected_model_paths:
            result.append("The model path is NOT matching with the default settings \n"
                          f"  default: {', '.join(expected_model_paths)} \n"
                          f"  current: {current_model_path}")
        else:
            folder_path = Path(current_model_path)
            if folder_path.exists() and folder_path.is_dir():
                matching_files = glob.glob(f"{folder_path}/model-00001*.safetensors")
                if matching_files:
                    md5sum = self.get_md5sum(matching_files[0])
                    if md5sum not in expected_model_md5sums:
                        result.append("The model's MD5 checksum does not match the expected value \n"
                                      f"  expected: {', '.join(expected_model_md5sums)} \n"
                                      f"  current:  {md5sum}")

        #validate dataset
        current_dataset_path = configs["harness_config"]["dataset_path"]
        expected_dataset_paths = benchmark["dataset_path"]
        expected_dataset_md5sums = benchmark["dataset_md5sum"]
        if current_dataset_path not in expected_dataset_paths:
            result.append("The dataset path is NOT matching with the default settings \n"
                          f"  default: {', '.join(expected_dataset_paths)} \n"
                          f"  current: {current_dataset_path}")
        else:
            dataset = Path(current_dataset_path)
            if dataset.exists() and dataset.is_file():
                md5sum = self.get_md5sum(dataset)
                if md5sum not in expected_dataset_md5sums:
                    result.append("The dataset's MD5 checksum does not match the expected value \n"
                                    f"  expected: {', '.join(expected_model_md5sums)} \n"
                                    f"  current:  {md5sum}")

        if result:
            print("Resource validation error:", file=sys.stderr)

            for value in result:
                print(value + "\n", file=sys.stderr)

            with open(invalid_file_path, "w") as file:
                for value in result:
                    file.write(value + "\n")

            if configs["harness_config"]["resource_checker_abort_on_failure"]:
                sys.exit(1)

    def get_md5sum(self, file):
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()
