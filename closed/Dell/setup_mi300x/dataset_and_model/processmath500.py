import sys, argparse
import pandas as pd
from transformers import AutoTokenizer


def main():

    parser = argparse.ArgumentParser(description="Convert math500 dataset to mlperf format")

    parser.add_argument('--model-path', type=str, required = True, help = 'Path to the model folder')
    parser.add_argument('--dataset-path', type=str, required = True, help = 'Path to the dataset file')
    parser.add_argument('--output-path', type=str, required = False, default = "math_500.pkl", help = 'Path to the output file')

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    try:
        df = pd.read_json(args.dataset_path, lines=True)
        
        tokenize = lambda s: tokenizer.encode(s)
        get_length = lambda t: len(t)  
        df["input"] = df["problem"]
        df["tok_input"] = df["input"].apply(tokenize)
        df["tok_input_length"] = df["tok_input"].apply(get_length)
        df["output"] = df["solution"]
        df["tok_output"] = df["output"].apply(tokenize)
        df["tok_output_length"] = df["tok_output"].apply(get_length)

        print(df.columns.tolist())
        df.to_pickle(args.output_path)
        print("Conversion finished")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
