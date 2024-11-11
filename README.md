# Loan Approval Classification Dataset
# 一、描述	
### 使用從 Kaggle 取得的貸款核准分類資料集		
來源可參考：[Loan Approval Classification Data on Kaggle  ](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)  

透過 Kaggle 上的貸款核准分類資料集，用監督式學習方法訓練資料集並進行資料的前處理及特徵工程。  
我們的目標是建立一個模型來預測貸款是否會被核准。

# 二、內容
欄位名稱:      
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
程式碼:  
```python
loan=pd.read_csv('loan_data.csv')
loan
```
執行結果:  
![螢幕擷取畫面 2024-11-11 142259](https://github.com/user-attachments/assets/c7eb3020-4b64-4063-bd6b-a39497b6723e)
該資料集包含 45,000 筆資料（rows）和 14 個欄位（columns）。  
這些欄位代表申請人和貸款申請的不同特徵，適合用於進行貸款核准的預測分析。

## 3.檢查數據類型和缺失值
程式碼:  
```python
print(loan.dtypes)
print(loan.isnull().sum())
```
執行結果:  
![螢幕擷取畫面 2024-11-11 144555](https://github.com/user-attachments/assets/e8167a51-3c15-41c7-a35d-7e2ca7e858a4)  
檢查資料型別和缺失值是為了確保資料格式正確並處理缺失值，以便進行分析。

## 4.數據編碼  
將性別、貸款違約紀錄、教育程度等類別進行編碼    
使用 pd.get_dummies 對房屋擁有狀況和貸款目的進行編碼  
程式碼:  
```python
#性別
loan['person_gender'] = loan['person_gender'].map({'female': 0, 'male': 1})

#貸款違約紀錄
loan['previous_loan_defaults_on_file'] = loan['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

#教育程度
education_mapping = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
loan['person_education'] = loan['person_education'].map(education_mapping)

#房屋擁有狀況
loan = pd.get_dummies(loan, columns=['person_home_ownership'], prefix='home_ownership')

#貸款目的
loan = pd.get_dummies(loan, columns=['loan_intent'], prefix='loan_intent')
```
進行編碼的原因是為了將類別型資料轉換為數值型資料，以便機器學習模型能夠理解並處理這些特徵。

## 5.確認型態
程式碼:  
```python
#原數字型態: ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'loan_status']
#原文字型態: ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

list1=[]
list2=[]
list3=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'loan_status']
for i in loan.columns:
    if loan[i].dtype=='object':
        try:
            loan[i]=loan[i].astype('float')
            list1.append(i)
        except:
            list2.append(i)
    else:
        list1.append(i)
print('數字型態:',list1)
print('文字型態:',list2)
print('原數字型態:',list3)
```
區分數值型和文字型欄位，並嘗試將可轉換的文字型欄位轉為數值型，是為了便於後續的資料清理、特徵工程和模型訓練。

## 6.欄位的中文說明
程式碼:  
```python
label={ 'person_age': '借款人的年齡',
        'person_gender': '借款人的性別',
        'person_education': '借款人的教育程度 高中 -> 0，專科 -> 1，學士 -> 2，碩士 -> 3，博士 -> 4。',
        'person_income': '借款人的年收入',
        'person_emp_exp': '借款人的工作經驗（年數）',
        
        'home_ownership_MORTGAGE': '抵押房產',
        'home_ownership_OTHER': '其他房產',
        'home_ownership_OWN': '自有房產',
        'home_ownership_RENT': '租房',

        'loan_amnt': '貸款金額',
        
        'loan_intent_DEBTCONSOLIDATION':'貸款用途:債務合併',
        'loan_intent_EDUCATION':'貸款用途:教育',
        'loan_intent_HOMEIMPROVEMENT':'貸款用途:家庭裝修',
        'loan_intent_MEDICAL':'貸款用途:醫療',
        'loan_intent_PERSONAL':'貸款用途:個人',
        'loan_intent_VENTURE':'貸款用途:創業',

        'loan_int_rate': '貸款利率（百分比）',
        'loan_percent_income': '貸款金額佔年收入的百分比',
        'cb_person_cred_hist_length': '信用歷史長度（年數）',
        'credit_score': '信用評分',
        'previous_loan_defaults_on_file': '是否有過往貸款違約記錄（No: 0, Yes: 1）',
        'loan_status': '貸款狀態（1: 批准, 0: 拒絕）'}
```
label 字典的作用是用來提供欄位的中文或詳細說明，便於在資料視覺化過程中顯示更易理解的標籤或標題。

## 7.繪製直方圖(list1欄位)
程式碼:  
```python
for cols in list1:
    print(label.get(cols)) 
    plt.title(cols) 
    sns.histplot(x=cols, hue='loan_status', kde=True, data=loan) 
    plt.show()
```
執行結果(以年齡為例):  
![image](https://github.com/user-attachments/assets/12297265-2836-4609-ab38-78621d96dc7f)

## 8.繪製箱型圖(list3原數字欄位)
程式碼:  
```python
for cols in list3:
    print(label.get(cols))
    plt.title(cols)
    sns.boxplot(x='loan_status', y=cols, data=loan)
    plt.show()
```
執行結果(以貸款金額為例):  
![image](https://github.com/user-attachments/assets/1ab7d392-ffdf-4309-b089-be6141b774db)

## 9.計算相關係數矩陣並繪製熱圖
程式碼:  
```python
print(loan.corr())
sns.heatmap(loan.corr())
```
執行結果:  
![output](https://github.com/user-attachments/assets/1958d42c-929c-4a13-9870-c49c6d558a6d)  
繪製熱力圖的目的是視覺化資料集中各個數值型特徵之間的相關性，以便更直觀地觀察變數間的關係。

## 10.列出相關係數
程式碼:  
```python
print(loan.corr()['loan_status'].sort_values(ascending=False))
```
執行結果:  
![螢幕擷取畫面 2024-11-11 151507](https://github.com/user-attachments/assets/dee6082d-27ac-4832-86d0-c1b1ab527a4d)

## 11.引入模組和工具用於資料分割、建模、模型評估、資料平衡及模型保存。
程式碼:  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN #過採樣
from imblearn.under_sampling import RandomUnderSampler,NearMiss #欠採樣
import joblib
```
## 12.隨機森林
使用隨機森林模型進行分類預測，並計算訓練和測試的準確率、分類報告和混淆矩陣。  
程式碼:  
```python
print('隨機森林模型')
X=loan.drop('loan_status',axis=1)
y=loan['loan_status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42) # 調整樹的數量 # 設定樹的最大深度

model.fit(X_train,y_train)

joblib.dump(model,'loan.pkl')
model2=joblib.load('loan.pkl')

predict1=model2.predict(X_test)
predict2=model2.predict(X_train)

print('train準確率:',accuracy_score(y_train,predict2))
print('test準確率:',accuracy_score(y_test,predict1))
print('分類報告:\n',classification_report(y_test,predict1))
print('混淆矩陣:\n',confusion_matrix(y_test,predict1))
```
執行結果:  
![image](https://github.com/user-attachments/assets/70b26af6-39a2-48c7-a31d-6452b856df11)

## 13.過採樣
使用 SMOTE 進行過採樣處理，以平衡數據集中的類別不平衡問題 。  
程式碼:  
```python
print('SMOTE 過採樣')

ros=SMOTE()
X_train_ros,y_train_ros=ros.fit_resample(X_train,y_train)
model.fit(X_train_ros,y_train_ros)

df1=pd.DataFrame({'feature1':X_train.iloc[:,0],'feature2':X_train.iloc[:,1],'loan_status':y_train})
print(df1['loan_status'].value_counts())
plt.show()

df1=pd.DataFrame({'feature1':X_train_ros.iloc[:,0],'feature2':X_train_ros.iloc[:,1],'loan_status':y_train_ros})
print(df1['loan_status'].value_counts())
plt.show()

model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42) # 調整樹的數量 # 設定樹的最大深度

model.fit(X_train_ros,y_train_ros)

joblib.dump(model,'loan_smote.pkl')
model2=joblib.load('loan_smote.pkl')

predict1=model2.predict(X_test)
predict2=model2.predict(X_train_ros)

print('train準確率:',accuracy_score(y_train_ros,predict2))
print('test準確率:',accuracy_score(y_test,predict1))
print('分類報告:\n',classification_report(y_test,predict1))
print('混淆矩陣:\n',confusion_matrix(y_test,predict1))
```
執行結果:  
![image](https://github.com/user-attachments/assets/b62dd030-78ff-4f0a-acb3-780358ad04bc)

# 三、結論
透過 Kaggle 的貸款核准資料集，利用監督式學習方法進行資料處理、特徵工程和模型建構。我們使用過採樣以處理類別不平衡問題，並選擇隨機森林分類器作為主要模型。結果顯示，透過SMOTE處理不平衡後，雖然精確率下降，但召回率有提高，可以依照自己的需求選擇處理方式。
