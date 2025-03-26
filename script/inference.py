import torch
from dataset.dataset import ielts_dataset, ielts_dataset_scaler
from dataset.preprocessing import preprocess
from torch.utils.data import DataLoader
import json
import os
from sklearn.metrics import mean_absolute_error

def inference_model(config, tokenizer, device):
    """
    Perform inference on the test data using the trained model.
    """
    # Load the model
    if config["model_type"] == "classification":
        from model.finetune_deberta_model import score_classifier as model_class
    elif config["model_type"] == "regression":
        from model.finetune_deberta_model import score_regression as model_class
    else:
        raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

    model = model_class(config["model_name"]).to(device)
    checkpoint_path = config["checkpoint_path"]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Prepare test dataset
    if config["model_type"] == "classification":
        test_dataset = ielts_dataset(config["test_data_path"])
    else:
        test_dataset = ielts_dataset_scaler(config["test_data_path"])

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: preprocess(data=x, tokenizer=tokenizer),
    )

    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)

            if config["model_type"] == "classification":
                output = model(data)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                pred = torch.argmax(logits, dim=1)
                pred = pred / 2.0
                true_labels = [label / 2.0 for label in true_labels]
            else:  # Regression
                output, _ = model(data)
                pred = output.squeeze()
                
                pred = torch.round(pred * 9)
                # print(pred)
                true_labels = [label * 9  for label in true_labels]
            
            
            predictions.extend(pred.cpu().tolist())
            true_labels.extend(target.cpu().tolist())
            print("label",true_labels) 
            print("pred",predictions)
    
    save_path = config["output_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mae = mean_absolute_error(true_labels, predictions)
    print(f"Mean Absolute Error (MAE): {mae}")

    save_path = config["output_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        # Write predictions
        for true, pred in zip(true_labels, predictions):
            f.write(f"{true},{pred}\n")
        # Write MAE at the end
        f.write(f"\nMAE,{mae}\n")

    print(f"Predictions and MAE saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./script/config.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from utils.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(config["model_name"])

    inference_model(config, tokenizer, device)
