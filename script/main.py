import argparse
import json
from utils.tokenizer import load_tokenizer
from script.train_with_margin import train_model
from script.inference import inference_model
# from script.evaluate import evaluate_model
from dataset.preprocessing import preprocess_data
from dataset.preprocessing import preprocess
from dataset.dataset import ielts_dataset
import torch
from utils.load_model import load_model
from model.finetune_deberta_model import score_classifier, score_regression

def main(args):
    with open(args.config_path) as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # preprocess_data(config["input_file"], config["output_file"])

    tokenizer = load_tokenizer(config["model_name"])

    if config["model_type"] == "classification":
        finetune_model = score_classifier(config["model_name"])
    elif config["model_type"] == "regression":
        finetune_model = score_regression(config["model_name"])
    
    if args.mode == 'train':
        train_model(args, config, tokenizer, finetune_model, device)
    elif args.mode == 'inference':
        inference_model(config, tokenizer, device)
    # elif args.mode == 'evaluate':
    #     evaluate_model(config, tokenizer, test_model, device)
    else:
        print("make sure your mode!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./script/config.json", help="Path to config file")
    parser.add_argument("--mode", choices=['train', 'evaluate','inference'], required=True, help="Choose mode")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", help="enter model's name.")
    parser.add_argument("--gpu", default=0, help="which gpu do you want to use", type=int)
    parser.add_argument("--start_checkpoint", default=10, help="start checkpoint", type=int)
    parser.add_argument("--end_checkpoint", default=20, help="last checkpoint", type=int)
    parser.add_argument("--save_interval", default=5, help="each step between checkpoint", type=int)
    parser.add_argument("--exp_name", default="exp_regression_1129_scaler_change", help="name your experiment", type=str)
    parser.add_argument("--contrastive_true", default=0)
    args = parser.parse_args()
    
    main(args)

# PYTHONPATH=$(pwd) python3 script/main.py --config_path ./script/config.json --mode train

    

    