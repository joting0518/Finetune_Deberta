import torch
from torch.utils.data import DataLoader
from dataset.dataset import ielts_dataset, ielts_dataset_scaler
from utils.optimizer import load_optimizer
import utils
from dataset.preprocessing import preprocess
from torch.utils.tensorboard import SummaryWriter
import logging, json, os
from utils.loss import supervised_contrastive_loss, SimCSE_loss, margin_loss

def train_model(args, config, tokenizer, finetune_model, device):
    EXP_FILE = args.exp_name
    MODEL_NAME = args.model_name
    model = finetune_model.to(device)
    optimizer = load_optimizer(
        lr=config["lr"],
        model_param=model.named_parameters(),
        weight_decay=config["weight_decay"],
    )
    writer = SummaryWriter(f"./exp/log/{EXP_FILE}")
    logging.basicConfig(filename=f"./exp/log/{EXP_FILE}/training.log", level=logging.DEBUG)
    assert MODEL_NAME is not None

    os.makedirs(f"./exp/{EXP_FILE}", exist_ok=True)
    os.makedirs(f"./exp/log/{EXP_FILE}", exist_ok=True)
    os.makedirs("./saved_model", exist_ok=True)

    with open(f"./exp/{EXP_FILE}/hyperparameters.json", "w") as f:
        f.write(json.dumps(args.__dict__, indent=4))

    if config["model_type"] == "classification":
        loss_fn = torch.nn.CrossEntropyLoss()
        dataset = ielts_dataset(config["train_data_path"])
        contrastive_loss_fn = (
            supervised_contrastive_loss if args.contrastive_true == 1 else SimCSE_loss
        )

        def model_forward_classification(data):
            score, score_positive, cls_feature, cls_feature_positive = model.forward_with_positive(data)
            contrastive_or_simCSE_loss = contrastive_loss_fn(
                cls_feature, 
                cls_feature_positive if contrastive_loss_fn == SimCSE_loss else target.view(-1),
                tau=0.1
            )
            basic_loss = loss_fn(score, target.long())
            # 要把target轉成1-20
            return score, contrastive_or_simCSE_loss, basic_loss

    else:
        loss_fn = torch.nn.MSELoss(reduction='mean')
        dataset = ielts_dataset_scaler(config["train_data_path"])

        def model_forward_regression(data):
            score, _ = model.forward(data)
            basic_loss = loss_fn(score, target)
            return score, 0, basic_loss

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: preprocess(data=x, tokenizer=tokenizer),
    )

    model.train()
    step = 0

    for epoch in range(config['num_epochs']):
        epoch_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if config["model_type"] == "classification":
                score, contrastive_or_simCSE_loss, basic_loss = model_forward_classification(data)
            else: 
                score, contrastive_or_simCSE_loss, basic_loss = model_forward_regression(data)

            print("target",target.size())
            print("score",score.size())
            margin_loss_value = margin_loss(score.squeeze(), target.squeeze())

            total_loss = basic_loss + margin_loss_value + contrastive_or_simCSE_loss
            total_loss.backward()
            optimizer.step()

            writer.add_scalar("Training/Loss", total_loss.item(), step)
            logging.info(f"Epoch {epoch}, Step {step}, Loss: {total_loss.item()}")

            epoch_loss += total_loss.item()
            step += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss}")
        writer.add_scalar("Epoch/Loss", avg_epoch_loss, epoch)

        if epoch >= args.start_checkpoint and epoch % args.save_interval == 0:
            checkpoint_path = f"./saved_model/{config['exp_name']}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model at epoch {epoch}: {checkpoint_path}")
            logging.info(f"Model saved at epoch {epoch}: {checkpoint_path}")

    checkpoint_path = f"./saved_model/{config['exp_name']}_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved final model: {checkpoint_path}")
    logging.info(f"Final model saved: {checkpoint_path}")

    print("Training complete.")
    writer.close()


# 用 clip 試試看
# 在 dataset就把分數標準化過了，在哪要轉回來？？loss function??或是？？ for regression
## mix loss function 是要放在？ regression


    
