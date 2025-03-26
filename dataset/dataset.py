import pandas as pd
import pickle
from torch.utils.data import Dataset
import numpy as np

def quantize_score(score):
    return int(np.round(score * 2))

class ielts_dataset_scaler(Dataset):
    def __init__(self, dataset_path) -> None:
        with open(dataset_path, 'r') as f:
            dataframe = pd.read_csv(f)

        self.essay = dataframe["Essay"].tolist()
        self.title = dataframe["Title"].tolist()


        # 將 Score 標準化到 [0, 1] 之間
        self.score = dataframe["Score"].tolist()
        max_score = max(self.score) 
        self.score = [s / max_score for s in self.score]  # 標準化

    def __getitem__(self, index):
        return self.essay[index], self.score[index], self.title[index]

    def __len__(self):
        return len(self.essay)
    
class ielts_dataset(Dataset):
    def __init__(self, dataset_path) -> None:
        with open(dataset_path, 'r') as f:
            dataframe = pd.read_csv(f)

        self.essay = dataframe["Essay"].tolist()
        self.title = dataframe["Title"].tolist()
        self.score = dataframe["Score"].tolist()
        self.quantized_score = [quantize_score(s) for s in self.score]
    def __getitem__(self, index):
        return self.essay[index], self.quantized_score[index], self.title[index]

    def __len__(self):
        return len(self.essay)
