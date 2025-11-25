# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("/income(1) (1).csv",na_values=[ " ?"])
 data
```
<img width="1291" height="405" alt="image" src="https://github.com/user-attachments/assets/62bffe93-3cac-4f1e-8291-9ab61e6ef714" />

```
 data.isnull().sum()
```
<img width="183" height="448" alt="image" src="https://github.com/user-attachments/assets/4535d421-2e1b-4bbc-9317-5ed396d9a5ff" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="1250" height="392" alt="image" src="https://github.com/user-attachments/assets/45d05a6e-9dfe-4daf-8662-494e92c3d85f" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1287" height="394" alt="image" src="https://github.com/user-attachments/assets/c41a002c-cc99-40b6-9cd6-eae5dee9c723" />

```
 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
<img width="314" height="205" alt="image" src="https://github.com/user-attachments/assets/c5f2ff08-38d0-4685-9c55-3b876b12a4e2" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="338" height="380" alt="image" src="https://github.com/user-attachments/assets/ec7a0fc0-dbc6-4144-97b6-eaf3e88e93f4" />

```
data2
```
<img width="1164" height="393" alt="image" src="https://github.com/user-attachments/assets/354f3fda-9454-4cd0-bc78-66f81ee10b53" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1494" height="437" alt="image" src="https://github.com/user-attachments/assets/f5579245-8327-4780-b924-874a8fc22029" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1482" height="47" alt="image" src="https://github.com/user-attachments/assets/46d86c65-5beb-452a-b2e6-73d97576e211" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1510" height="26" alt="image" src="https://github.com/user-attachments/assets/35da6840-bfdc-4aca-8cd4-04f52f728ec4" />

```
 y=new_data['SalStat'].values
 print(y)
```
<img width="163" height="26" alt="image" src="https://github.com/user-attachments/assets/771fc934-b070-42a3-8182-239b413675b4" />

```
x=new_data[features].values
print(x)
```
<img width="315" height="129" alt="image" src="https://github.com/user-attachments/assets/6f33b0cb-cd0f-4341-83be-1ef6c6b970d3" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="243" height="59" alt="image" src="https://github.com/user-attachments/assets/4bb95287-3364-4ac2-bf4f-f610dc465c85" />

```
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="121" height="69" alt="image" src="https://github.com/user-attachments/assets/2c9aae31-c27f-4afc-a483-81f5f189f824" />

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```
<img width="160" height="31" alt="image" src="https://github.com/user-attachments/assets/394de08a-e276-4657-a782-540acce0ce90" />

```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="222" height="28" alt="image" src="https://github.com/user-attachments/assets/9274e91f-9b1a-40b9-896f-44db1475fa41" />

```
data.shape
```
<img width="86" height="43" alt="image" src="https://github.com/user-attachments/assets/28b7294e-063b-48eb-b2f7-dee4567d141c" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']].values.ravel()
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="268" height="67" alt="image" src="https://github.com/user-attachments/assets/61fb9b8b-d5cc-4059-89cf-78770502c1c0" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="398" height="200" alt="image" src="https://github.com/user-attachments/assets/364741ad-5475-4dee-991d-d0e7dd591a02" />

```
 tips.time.unique()
```
<img width="359" height="53" alt="image" src="https://github.com/user-attachments/assets/06be88db-98e6-4b3a-8db6-f85d9377c2c5" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="197" height="85" alt="image" src="https://github.com/user-attachments/assets/eef76aaa-2620-4ab1-be31-3b9b4e304d56" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="342" height="44" alt="image" src="https://github.com/user-attachments/assets/d06e9176-ed7c-4988-a894-fb3696ee015c" />
     
# RESULT:
 Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
 save the data to a file is been executed.
