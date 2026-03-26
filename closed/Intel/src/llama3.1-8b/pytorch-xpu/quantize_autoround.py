from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from auto_round import AutoRound
from datasets import Dataset
from llmcompressor.transformers import oneshot
import os
import pandas as pd
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="facebook/opt-125m"
    )
    parser.add_argument(
        "--dataset", type=str, required=True
    )
    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")
    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")
    parser.add_argument("--sym", default=False, action='store_true',
                        help=" sym quantization")
    parser.add_argument("--iters", default=200, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")
    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")
    parser.add_argument("--device", default='fake', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu', 'gpu' and 'xpu'."
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Note: Using pickle with trusted dataset files only
    # In production, consider using safer serialization formats like JSON or HDF5
    dataframe = pd.read_pickle(args.dataset)  # nosec B301
    dataframe_str = dataframe["input"]
    dataset_list = []
    token_list = []
    for data_str in dataframe_str:
        data_token = tokenizer.encode(
            data_str,
            padding="max_length",
            max_length=1024,
            padding_side="right")
        token_list.append(data_token)
        data_str = tokenizer.decode(data_token)
        dataset_list.append(data_str)

    ds = Dataset.from_dict({"input_ids": token_list})
    recipe = """
    quant_stage:
      quant_modifiers:
        QuantizationModifier:
          kv_cache_scheme:
            num_bits: 8
            type: int
            strategy: tensor
            dynamic: false
            symmetric: true
    """
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        shuffle_calibration_samples=False,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )


    autoround = AutoRound(
        model,
        tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        iters=args.iters,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        nsamples=len(dataframe_str),
        seqlen=1024,
        batch_size=args.batch_size,
        dataset=dataset_list,
        enable_torch_compile=False,
        amp=False,
        device=args.device
        )
    
    packing_format="gptq"
    orig_path = args.model_name
    if orig_path.endswith("/"):
        output_dir=orig_path[:-1]+f"-{packing_format}-w{args.bits}g{args.group_size}kv"
    else:
        output_dir=orig_path+f"-{packing_format}-w{args.bits}g{args.group_size}kv"
    autoround.quantize_and_save(output_dir, format=f'auto_{packing_format}')

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'r') as file:
        data = json.load(file)
        data["quantization_config"] = model.config.quantization_config
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=2)

    pass

if __name__ == "__main__":
    main()
