import torch
import torch.nn as nn
import transformers

# cross entropy has softmax, so we just put original logit into loss function
# but we also need to pass actural result to user, so we do softmax(logit) for inference.
class score_classifier(nn.Module):
    def __init__(
        self, model_name: str = "microsoft/deberta-v3-base", drop: float = 0.0, score: int = 20
    ):

        super(score_classifier, self).__init__()

        self.pretrained_model = transformers.DebertaV2Model.from_pretrained(model_name)

        self.scorer = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.Dropout(p = drop, inplace=False), 
            nn.Linear(in_features=768, out_features=score),
        )

    def forward(self, batch_x):

        model_predict = self.pretrained_model(**batch_x)

        cls_feature = model_predict.last_hidden_state[:, 0, :]
        
        score = self.scorer(cls_feature)

        return score, cls_feature
    
    # 允許模型對同一樣本多次預測(?)
    def forward_with_positive(self, batch_x):

        model_predict = self.pretrained_model(**batch_x)
        model_predict_positive = self.pretrained_model(**batch_x)

        cls_feature = model_predict.last_hidden_state[:, 0, :]
        cls_feature_positive = model_predict_positive.last_hidden_state[:, 0, :]
        
        score = self.scorer(cls_feature)
        score_positive = self.scorer(cls_feature_positive)

        return score, score_positive, cls_feature, cls_feature_positive


class score_regression(nn.Module):
    def __init__(
        self, model_name: str = "microsoft/deberta-v3-base", drop: float = 0.0, token_num = 0
    ):

        super(score_regression, self).__init__()
        # 調整 vocabulary list 的大小(token -> vocabulary + specific prompt's token)
        # self.pretrained_model = transformers.BertModel.from_pretrained(model_name)
        # self.pretrained_model.resize_token_embeddings(token_num)
        self.pretrained_model = transformers.DebertaV2Model.from_pretrained(model_name)

        self.scorer = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.Dropout(p = drop, inplace=False), 
            nn.Linear(in_features=768, out_features=1),
        )
# 預測0-1的效果比0-9好，所以嘗試用sigmoid以及前處理就先用scaler分數到0-1之間
    def forward(self, batch_x):

        model_predict = self.pretrained_model(**batch_x)
        cls_feature = model_predict.last_hidden_state[:, 0, :]
        
        # score = torch.sigmoid(self.scorer(cls_feature))
        score = self.scorer(cls_feature)

        return score, cls_feature
