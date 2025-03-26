import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import transformers
from transformers import DebertaV2TokenizerFast

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['Essay', 'Score'])
    df['Score'] = df['Score'].astype(float)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(output_path.replace('.csv', '_train.csv'), index=False)
    test_df.to_csv(output_path.replace('.csv', '_test.csv'), index=False)

    print("split data done!")

def preprocess(data, tokenizer: transformers.DebertaV2TokenizerFast):
    # print(data[0])
    # 文章（x[0]）和標題（x[1]），concat
    print(f"title: {data[0][2]}")
    print(f"score: {data[0][1]}")
    print(f"essay: {data[0][0]}")
    texts = [f"Title: {x[2]} Essay: {x[0]}" for x in data]  # format：Title: Essay

    tokenized_str = tokenizer(
        texts,
        truncation=True,          
        max_length=512,           
        padding="longest",      
        return_tensors="pt",
    )

    # 分數（x[2]）
    scores = torch.FloatTensor([float(x[1]) for x in data])
    print("preprocessing done!")
    return tokenized_str, scores
