# AES: Automatic Essay Scoring using DeBERTa

This project focuses on **Automatic Essay Scoring (AES)**, particularly tailored for **IELTS writing tasks**. We fine-tune a DeBERTa model to assess writing quality either as a **classification** or **regression** task.

## 🧠 Core Idea

Using pretrained language models like `microsoft/deberta-v3-base`, we fine-tune on IELTS writing data to predict human-like writing scores.

## 📊 Experiment: Regression vs Classification in AES

During the fine-tuning of the DeBERTa language model for AES (Automatic Essay Scoring), we experimented with two different objective functions: **regression** and **classification**.

### 📉 Regression Training Results

As shown in the figure below, when training the model with a **regression objective**, the loss failed to converge properly and continued to fluctuate over time.

![Regression Loss Trends](./images/regression_loss.png)

*Figure 3. Regression task training loss trends.*

Further evaluation showed that the model's predicted average score was 6.0. Upon analysis, we found this was due to **imbalanced data distribution**, which made it difficult for the model to learn meaningful score representations through regression. The essay score distribution is shown below:

![Score Distribution](./images/score_distribution.png)

*Figure 4. Histogram of score distribution (Total essays: 3,676).*

### ✅ Classification Training Results

To address the convergence issue, we re-formulated the problem as a **classification task**. Scores were mapped into 19 discrete classes, and the model was trained using **Cross-Entropy loss**.

This approach allowed the model to learn more robustly from the data and converge successfully, as illustrated in the following loss curves:

![Classification Loss Trends](./images/classification_loss.png)

*Figure 5. Classification task training loss trends.*

Final evaluation results show that the model's predicted scores closely match human-labeled scores, with a **Mean Absolute Error (MAE) of 0.7064**, which is approximately within ±0.5 band of the true IELTS scoring range.

---

**Conclusion:**  
Formulating AES as a **classification** task provides significantly better training stability and scoring accuracy compared to regression, especially when working with imbalanced writing score distributions.

### file structure
writing_score_model/
│
├── dataset/                        #資料處理相關
│   ├── __init__.py
│   ├── dataset.py                  #Dataset處理程式碼
│   └── preprocessing.py            #資料清理和預處理程式
│
├── model/                          #模型定義
│   ├── __init__.py
│   └── deberta_model.py            #微調的 DeBERTa 模型定義
│
├── script/                         #訓練和測試腳本
│   ├── __init__.py
│   ├── train.py                    #模型訓練程式
│   ├── evaluate.py                 #模型評估程式
│   ├── config.json                 #訓練參數的配置檔案
│   └── main.py                     #主程式入口，負責調度整個流程
│
├── util/                           #工具函數 (loss, optimizer, scheduler)
│   ├── __init__.py
│   ├── load_model.py               #模型載入函數
│   ├── loss.py                     #定義損失函數
│   ├── optimizer.py                #優化器設置
│   ├── scheduler.py                #Scheduler 設置
│   └── tokenizer.py                #Tokenizer 初始化
│
├── logs/                           #訓練日誌
│
├── saved_model/                    #已訓練好的模型
│
├── preprocess_dataset_test.csv     #寫作評分測試集
└── preprocess_dataset_train.csv    #寫作評分訓練集
