import torch
def load_model(model, exp_name: str, checkpoint: int):

    checkpoint_path = f"./saved_model/{exp_name}_ckpt_{checkpoint}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    print(f"Loaded model from {checkpoint_path}")
    return model