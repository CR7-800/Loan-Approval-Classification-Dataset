# Loan Approval Classification Dataset
# 一、說明	
### 使用從 Kaggle 取得的貸款核准分類資料集		
來源可參考：[Loan Approval Classification Data on Kaggle  ](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)  

本專案透過 Kaggle 上的貸款核准分類資料集，用監督式學習方法訓練資料集並進行資料的前處理及特徵工程。我們的目標是建立一個模型來預測貸款是否會被核准。專案中我們進行了探索性資料分析（EDA）、資料清理、特徵選取、模型訓練與測試，以驗證模型的預測準確性。

# 二、內容
欄位名稱    
![螢幕擷取畫面 2024-11-11 135846](https://github.com/user-attachments/assets/fe4edfd7-16c1-4f65-80ff-e92c1856a279)

## 1.安裝與引入模組 
在開始資料分析前，請確保已安裝 pandas、matplotlib 和 seaborn 模組。可以使用以下指令安裝：

```bash
pip install pandas
pip install matplotlib
pip install seaborn
```
程式碼:  
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
pandas 用於資料處理，matplotlib.pyplot 和 seaborn 用於資料視覺化。

## 2.使用 pandas 讀取 CSV 文件
```python
loan=pd.read_csv('loan_data.csv')
loan.head()
```
執行結果:  
![螢幕擷取畫面 2024-11-11 142259](https://github.com/user-attachments/assets/c7eb3020-4b64-4063-bd6b-a39497b6723e)
