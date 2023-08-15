```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_predict, TimeSeriesSplit
from keras.utils import to_categorical 
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, auc, precision_score, recall_score, precision_recall_curve, roc_curve, f1_score, fbeta_score
from keras.layers import Dense
from keras.models import Sequential 
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LassoCV
```

    D:\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
pd.set_option('display.max_columns', 500)
df = pd.read_csv("default of credit card clients.csv")
df=df.rename(columns={'default payment next month':'default_next_month','PAY_0':'PAY_1'})
df=df.drop('ID',axis=1)
data=df.copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default_next_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# EDA and organic growth


```python
df1 = df.copy()
```


```python
df1.shape
```




    (30000, 24)



import pandas_profiling

pandas_profiling.ProfileReport(df).to_file("profile_report.html")


```python
df1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default_next_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>167484.322667</td>
      <td>1.603733</td>
      <td>1.853133</td>
      <td>1.551867</td>
      <td>35.485500</td>
      <td>-0.016700</td>
      <td>-0.133767</td>
      <td>-0.166200</td>
      <td>-0.220667</td>
      <td>-0.266200</td>
      <td>-0.291100</td>
      <td>51223.330900</td>
      <td>49179.075167</td>
      <td>4.701315e+04</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
      <td>5663.580500</td>
      <td>5.921163e+03</td>
      <td>5225.68150</td>
      <td>4826.076867</td>
      <td>4799.387633</td>
      <td>5215.502567</td>
      <td>0.221200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129747.661567</td>
      <td>0.489129</td>
      <td>0.790349</td>
      <td>0.521970</td>
      <td>9.217904</td>
      <td>1.123802</td>
      <td>1.197186</td>
      <td>1.196868</td>
      <td>1.169139</td>
      <td>1.133187</td>
      <td>1.149988</td>
      <td>73635.860576</td>
      <td>71173.768783</td>
      <td>6.934939e+04</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
      <td>16563.280354</td>
      <td>2.304087e+04</td>
      <td>17606.96147</td>
      <td>15666.159744</td>
      <td>15278.305679</td>
      <td>17777.465775</td>
      <td>0.415062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-165580.000000</td>
      <td>-69777.000000</td>
      <td>-1.572640e+05</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>3558.750000</td>
      <td>2984.750000</td>
      <td>2.666250e+03</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
      <td>1000.000000</td>
      <td>8.330000e+02</td>
      <td>390.00000</td>
      <td>296.000000</td>
      <td>252.500000</td>
      <td>117.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>140000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22381.500000</td>
      <td>21200.000000</td>
      <td>2.008850e+04</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
      <td>2100.000000</td>
      <td>2.009000e+03</td>
      <td>1800.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>240000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>67091.000000</td>
      <td>64006.250000</td>
      <td>6.016475e+04</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
      <td>5006.000000</td>
      <td>5.000000e+03</td>
      <td>4505.00000</td>
      <td>4013.250000</td>
      <td>4031.500000</td>
      <td>4000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1000000.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>964511.000000</td>
      <td>983931.000000</td>
      <td>1.664089e+06</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
      <td>873552.000000</td>
      <td>1.684259e+06</td>
      <td>896040.00000</td>
      <td>621000.000000</td>
      <td>426529.000000</td>
      <td>528666.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1['EDUCATION'].unique().tolist()
```




    [2, 1, 3, 5, 4, 6, 0]




```python
for i in ['SEX','MARRIAGE','EDUCATION']:
    print (str(i)+':'+str(sorted(df1[i].unique().tolist())))
```

    SEX:[1, 2]
    MARRIAGE:[0, 1, 2, 3]
    EDUCATION:[0, 1, 2, 3, 4, 5, 6]
    

Labels 4,5 & 6 are classified as either unknown or 'others' for the EDUCATION column, therefore could be regrouped under a single label denoting 'others'

Similarly, the variable MARRIAGE has the label 0 as unknown


```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), cmap='rocket_r', annot=True)
plt.title("Correlation matrix", fontsize=18)
plt.savefig('corr1.png',transparent=True)
```


    
![png](output_10_0.png)
    



```python
plt.figure(figsize=(15,10))
sns.countplot(data=df1, x='AGE', hue='default_next_month', dodge=False)
sns.despine()
plt.title("Age distribution", fontsize=18)
plt.savefig("age_def.png", transparent= True)
```


    
![png](output_11_0.png)
    



```python
plt.figure(figsize=(15,10))
sns.countplot(data=df1, x='AGE', dodge=False)
sns.despine()
plt.title("Age distribution", fontsize=18)
plt.savefig("age.png", transparent= True)
```


    
![png](output_12_0.png)
    



```python
print(df.AGE.max())
print(df.AGE.min())
print(df.AGE.mean())
```

    79
    21
    35.4855
    


```python
df_age=pd.DataFrame(df1['AGE'].loc[df1['default_next_month'] == 0].value_counts())
df_age.columns=['no_default']
df_age['default']=df1['AGE'].loc[df1['default_next_month'] == 1].value_counts()
df_age['Total']=df1['AGE'].value_counts()
df_age['%no_default']=(df_age['no_default']/df_age['Total'])*100
df_age['%default']=(df_age['default']/df_age['Total'])*100
df_age=df_age.rename_axis('Age').reset_index()
df_age=df_age.drop(['no_default','default','Total'],axis=1)
df_age=df_age.fillna(0)
df_age=df_age.sort_values(by=['Age'],ascending=False)
```


```python
ax=df_age.plot(figsize=(10,10),kind='barh',x='Age', stacked=True, fontsize=13)
ax.set_title("Probability of default by age", fontsize=18)
ax.set_ylabel("Age", fontsize=18)
ax.set_xlabel("Percentage", fontsize=18)
plt.savefig("age_proba.png", transparent=True)
```


    
![png](output_15_0.png)
    



```python
plt.figure(figsize=(10,10))
plt.pie(x=df1['default_next_month'].value_counts(), labels= ['No','Yes'], autopct='%1.1f%%', colors=['coral','deepskyblue'])
plt.title("Default next month?", fontsize=18)
plt.savefig("default_prop.png", transparent=True)
```


    
![png](output_16_0.png)
    



```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state=37)
kmeans.fit(pd.DataFrame(df1['AGE']))
y_kmeans = kmeans.predict(pd.DataFrame(df1['AGE']))
df1['age_cluster']=y_kmeans
```


```python
print('cluster0:'+str(df1.loc[df1['age_cluster']==0].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==0].AGE.max()))
df1.loc[df1['age_cluster']==0,'age_cluster']=(str(df1.loc[df1['age_cluster']==0].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==0].AGE.max()))

print('cluster1:'+str(df1.loc[df1['age_cluster']==1].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==1].AGE.max()))
df1.loc[df1['age_cluster']==1,'age_cluster']=(str(df1.loc[df1['age_cluster']==1].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==1].AGE.max()))

print('cluster2:'+str(df1.loc[df1['age_cluster']==2].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==2].AGE.max()))
df1.loc[df1['age_cluster']==2,'age_cluster']=(str(df1.loc[df1['age_cluster']==2].AGE.min())+'-'+str(df1.loc[df1['age_cluster']==2].AGE.max()))
```

    cluster0:33-44
    cluster1:45-79
    cluster2:21-32
    


```python
P_default_age=df1.loc[df1['default_next_month']==1].age_cluster.value_counts()/df1.age_cluster.value_counts()
P_default_age
```




    21-32    0.218479
    33-44    0.213403
    45-79    0.244798
    Name: age_cluster, dtype: float64



# Education


```python
df1.loc[(df['EDUCATION']==5) | (df['EDUCATION']==6) | (df['EDUCATION']==0),'EDUCATION']=4
```


```python
edu={1:'Graduate School',2:'University',3:'High School',4:'Others'}
for i in edu:
    df1.loc[df1['EDUCATION']==i, 'EDUCATION']=edu[i]
```


```python
df1['EDUCATION'].value_counts()
```




    University         14030
    Graduate School    10585
    High School         4917
    Others               468
    Name: EDUCATION, dtype: int64




```python
plt.figure(figsize=(10,10))
plt.pie(x=df1['EDUCATION'].value_counts(), labels= ['University','Graduate School','High School','Others'], autopct='%1.1f%%', colors=['coral','deepskyblue', 'orange','yellowgreen'])
plt.title("Education distribution", fontsize=18)
plt.savefig("edu_prop.png", transparent=True)
```


    
![png](output_24_0.png)
    



```python
P_default_edu=df1.loc[df1['default_next_month']==1].EDUCATION.value_counts()/df1.EDUCATION.value_counts()
P_default_edu
```




    University         0.237349
    Graduate School    0.192348
    High School        0.251576
    Others             0.070513
    Name: EDUCATION, dtype: float64




```python
df_edu=pd.DataFrame(df1['EDUCATION'].loc[df1['default_next_month'] == 0].value_counts())
df_edu.columns=['no_default']
df_edu['default']=df1['EDUCATION'].loc[df1['default_next_month'] == 1].value_counts()
df_edu['Total']=df1['EDUCATION'].value_counts()
df_edu['%no_default']=(df_edu['no_default']/df_edu['Total'])*100
df_edu['%default']=(df_edu['default']/df_edu['Total'])*100
df_edu=df_edu.rename_axis('Level of Education').reset_index()
df_edu=df_edu.drop(['no_default','default','Total'],axis=1)
df_edu

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Level of Education</th>
      <th>%no_default</th>
      <th>%default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>University</td>
      <td>76.265146</td>
      <td>23.734854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Graduate School</td>
      <td>80.765234</td>
      <td>19.234766</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High School</td>
      <td>74.842384</td>
      <td>25.157616</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Others</td>
      <td>92.948718</td>
      <td>7.051282</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax=df_edu.plot(figsize=(10,10),kind='bar',x='Level of Education', stacked=True, fontsize=13)
ax.set_title("Probability of default by education level", fontsize=18)
ax.set_ylabel("Percentage", fontsize=18)
ax.set_xticklabels(rotation=45, labels=df_edu['Level of Education'].to_list())
plt.savefig("edu_proba.png", transparent=True)
```


    
![png](output_27_0.png)
    


High school education highest % of default;
Others lowest

# Sex


```python
gender={1:'Male',2:'Female'}
for i in gender:
    df1.loc[df1['SEX']==i, 'SEX']=gender[i]
```


```python
df1['SEX'].value_counts()
```




    Female    18112
    Male      11888
    Name: SEX, dtype: int64




```python
plt.figure(figsize=(10,10))
plt.pie(x=df1['SEX'].value_counts(), labels= ['Female','Male'], autopct='%1.1f%%', colors=['coral','deepskyblue'])
plt.title('Gender distribution', fontsize=18)
plt.savefig("gender_prop.png", transparent=True)
```


    
![png](output_32_0.png)
    



```python
df_sex=pd.DataFrame(df1['SEX'].loc[df1['default_next_month'] == 0].value_counts())
df_sex.columns=['no_default']
df_sex['default']=df1['SEX'].loc[df1['default_next_month'] == 1].value_counts()
df_sex['Total']=df1['SEX'].value_counts()
df_sex['%no_default']=(df_sex['no_default']/df_sex['Total'])*100
df_sex['%default']=(df_sex['default']/df_sex['Total'])*100
df_sex=df_sex.rename_axis('Gender').reset_index()
df_sex=df_sex.drop(['no_default','default','Total'],axis=1)
df_sex
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>%no_default</th>
      <th>%default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>79.223719</td>
      <td>20.776281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>75.832773</td>
      <td>24.167227</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax=df_sex.plot(figsize=(10,10),kind='bar',x='Gender', stacked=True, fontsize=13)
ax.set_title("Probability of default by gender", fontsize=18)
ax.set_ylabel("Percentage", fontsize=18)
ax.set_xticklabels(rotation=45, labels=df_sex['Gender'].to_list())
plt.savefig("gender_proba.png", transparent=True)
```


    
![png](output_34_0.png)
    


# Relationships status


```python
df1['MARRIAGE'].value_counts()
```




    2    15964
    1    13659
    3      323
    0       54
    Name: MARRIAGE, dtype: int64




```python
plt.figure(figsize=(10,10))
ax=sns.violinplot(data=df1, x='MARRIAGE',y='AGE')
ax.set_title("Age distribution by marital status", fontsize=18)
plt.savefig("marriage_age.png", transparent=True)
```


    
![png](output_37_0.png)
    



```python
plt.figure(figsize=(10,10))
sns.violinplot(data=df1[df1['MARRIAGE']!=0], x='MARRIAGE',y='AGE', hue='SEX')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f66c14e6d8>




    
![png](output_38_1.png)
    


Men tend to stay single for longer and get married at an older age


```python
relation={0:'Others',1:'Married',2:'Single',3:'Divorced'}
for i in relation:
    df1.loc[df1['MARRIAGE']==i, 'MARRIAGE']=relation[i]
```


```python
df_marital=pd.DataFrame(df1['MARRIAGE'].loc[df1['default_next_month'] == 0].value_counts())
df_marital.columns=['no_default']
df_marital['default']=df1['MARRIAGE'].loc[df1['default_next_month'] == 1].value_counts()
df_marital['Total']=df1['MARRIAGE'].value_counts()
df_marital['%no_default']=(df_marital['no_default']/df_marital['Total'])*100
df_marital['%default']=(df_marital['default']/df_marital['Total'])*100
df_marital=df_marital.rename_axis('Marital status').reset_index()
df_marital=df_marital.drop(['no_default','default','Total'],axis=1)
df_marital
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>%no_default</th>
      <th>%default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Single</td>
      <td>79.071661</td>
      <td>20.928339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Married</td>
      <td>76.528296</td>
      <td>23.471704</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Divorced</td>
      <td>73.993808</td>
      <td>26.006192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Others</td>
      <td>90.740741</td>
      <td>9.259259</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax=df_marital.plot(figsize=(10,10),kind='bar',x='Marital status', stacked=True, fontsize=13)
ax.set_title("Probability of default by marital status", fontsize=18)
ax.set_ylabel("Percentage", fontsize=18)
ax.set_xticklabels(rotation=45, labels=df_marital['Marital status'].to_list())
plt.savefig("marriage_proba.png",transparent=True)
```


    
![png](output_42_0.png)
    


# Bill amount


```python
df.BILL_AMT1.max()
```




    964511




```python
df.LIMIT_BAL.max()
```




    1000000




```python
a=pd.DataFrame(df[df['AGE']!=79].groupby('AGE').mean().BILL_AMT1).reset_index()
plt.figure(figsize=(15,10))
sns.barplot(data=a,x='AGE',y='BILL_AMT1', palette='rocket_r')
plt.title("Bill statement amount by age", fontsize=18)
plt.savefig("BILL.png",transparent=True)
```


    
![png](output_46_0.png)
    



```python
df.BILL_AMT1.value_counts().count()
```




    22723




```python
df[df['AGE']>=75][['AGE','BILL_AMT1']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>BILL_AMT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>246</th>
      <td>75</td>
      <td>52874</td>
    </tr>
    <tr>
      <th>18245</th>
      <td>79</td>
      <td>429309</td>
    </tr>
    <tr>
      <th>25136</th>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25141</th>
      <td>75</td>
      <td>205601</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df['BILL_AMT1']<0)].shape
```




    (590, 24)




```python
df[(df['BILL_AMT1']<0)&(df['default_next_month']==1)].shape
```




    (109, 24)



# PAY_AMT


```python
b=pd.DataFrame(df[df['AGE']!=79].groupby('AGE').mean().PAY_AMT1).reset_index()
plt.figure(figsize=(15,10))
sns.barplot(data=b,x='AGE',y='PAY_AMT1', palette='Blues')
plt.title("Payment amount by age", fontsize=18)
plt.savefig("PAY.png",transparent=True)
```


    
![png](output_52_0.png)
    



```python
df[(df['PAY_AMT1']>df['BILL_AMT2'])].shape
```




    (3464, 24)




```python
#This shows that multiple clients (417) overpaid last month's bill and still defaulted
df[(df['PAY_AMT1']>df['BILL_AMT2'])&(df['default_next_month']==1)].shape
```




    (417, 24)




```python
# 620 out of the 2500 with negative bill amounts overpaid, therefore the negative bill amounts are not explained by this
df[(df['PAY_AMT1']>df['BILL_AMT2'])&(df['BILL_AMT1']<0)].shape
```




    (588, 24)




```python
df[(df['PAY_AMT1']>df['BILL_AMT2'])&(df['BILL_AMT1']<0)&(df['default_next_month']==1)].shape
```




    (108, 24)




```python
df[(df['BILL_AMT1']<=0)&(df['default_next_month']==1)].shape
```




    (643, 24)



# Credit Limit


```python
plt.figure(figsize=(15,10))
ax=sns.countplot(data=df.sort_values(by=['LIMIT_BAL'],ascending=False),x='LIMIT_BAL')
```


    
![png](output_59_0.png)
    



```python
df[df['LIMIT_BAL']!=0].LIMIT_BAL.mode()
```




    0    50000
    dtype: int64




```python
df_lim=pd.DataFrame(df1['LIMIT_BAL'].loc[df1['default_next_month'] == 0].value_counts())
df_lim.columns=['no_default']
df_lim['default']=df1['LIMIT_BAL'].loc[df1['default_next_month'] == 1].value_counts()
df_lim['Total']=df1['LIMIT_BAL'].value_counts()
df_lim['%no_default']=(df_lim['no_default']/df_lim['Total'])*100
df_lim['%default']=(df_lim['default']/df_lim['Total'])*100
df_lim=df_lim.rename_axis('Credit Limit').reset_index()
df_lim=df_lim.drop(['no_default','default','Total'],axis=1)
df_lim=df_lim.sort_values(by=['Credit Limit'],ascending=False)
```


```python
ax=df_lim.plot(figsize=(10,10),kind='barh',x='Credit Limit', stacked=True, fontsize=13)
ax.set_title("Probability of default by credit limit", fontsize=18)
ax.set_ylabel("Credit Limit", fontsize=18)
ax.set_xlabel("Percentage", fontsize=18)
ax.set_yticklabels(labels=df_lim['Credit Limit'],fontsize=8)
```




    [Text(0,0,'1000000'),
     Text(0,0,'800000'),
     Text(0,0,'780000'),
     Text(0,0,'760000'),
     Text(0,0,'750000'),
     Text(0,0,'740000'),
     Text(0,0,'730000'),
     Text(0,0,'720000'),
     Text(0,0,'710000'),
     Text(0,0,'700000'),
     Text(0,0,'690000'),
     Text(0,0,'680000'),
     Text(0,0,'670000'),
     Text(0,0,'660000'),
     Text(0,0,'650000'),
     Text(0,0,'640000'),
     Text(0,0,'630000'),
     Text(0,0,'620000'),
     Text(0,0,'610000'),
     Text(0,0,'600000'),
     Text(0,0,'590000'),
     Text(0,0,'580000'),
     Text(0,0,'570000'),
     Text(0,0,'560000'),
     Text(0,0,'550000'),
     Text(0,0,'540000'),
     Text(0,0,'530000'),
     Text(0,0,'520000'),
     Text(0,0,'510000'),
     Text(0,0,'500000'),
     Text(0,0,'490000'),
     Text(0,0,'480000'),
     Text(0,0,'470000'),
     Text(0,0,'460000'),
     Text(0,0,'450000'),
     Text(0,0,'440000'),
     Text(0,0,'430000'),
     Text(0,0,'420000'),
     Text(0,0,'410000'),
     Text(0,0,'400000'),
     Text(0,0,'390000'),
     Text(0,0,'380000'),
     Text(0,0,'370000'),
     Text(0,0,'360000'),
     Text(0,0,'350000'),
     Text(0,0,'340000'),
     Text(0,0,'330000'),
     Text(0,0,'320000'),
     Text(0,0,'310000'),
     Text(0,0,'300000'),
     Text(0,0,'290000'),
     Text(0,0,'280000'),
     Text(0,0,'270000'),
     Text(0,0,'260000'),
     Text(0,0,'250000'),
     Text(0,0,'240000'),
     Text(0,0,'230000'),
     Text(0,0,'220000'),
     Text(0,0,'210000'),
     Text(0,0,'200000'),
     Text(0,0,'190000'),
     Text(0,0,'180000'),
     Text(0,0,'170000'),
     Text(0,0,'160000'),
     Text(0,0,'150000'),
     Text(0,0,'140000'),
     Text(0,0,'130000'),
     Text(0,0,'120000'),
     Text(0,0,'110000'),
     Text(0,0,'100000'),
     Text(0,0,'90000'),
     Text(0,0,'80000'),
     Text(0,0,'70000'),
     Text(0,0,'60000'),
     Text(0,0,'50000'),
     Text(0,0,'40000'),
     Text(0,0,'30000'),
     Text(0,0,'20000'),
     Text(0,0,'16000'),
     Text(0,0,'10000')]




    
![png](output_62_1.png)
    



```python
L=[]
for i in list(range(1,7)):
    x='%REMAINING_BAL'+str(i)
    L.append(x)
    y='BILL_AMT'+str(i)
    df[x]=(df['LIMIT_BAL']-df[y])/df['LIMIT_BAL']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default_next_month</th>
      <th>%REMAINING_BAL1</th>
      <th>%REMAINING_BAL2</th>
      <th>%REMAINING_BAL3</th>
      <th>%REMAINING_BAL4</th>
      <th>%REMAINING_BAL5</th>
      <th>%REMAINING_BAL6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.804350</td>
      <td>0.844900</td>
      <td>0.965550</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
      <td>0.977650</td>
      <td>0.985625</td>
      <td>0.977650</td>
      <td>0.972733</td>
      <td>0.971208</td>
      <td>0.972825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
      <td>0.675122</td>
      <td>0.844144</td>
      <td>0.849344</td>
      <td>0.840767</td>
      <td>0.833911</td>
      <td>0.827233</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
      <td>0.060200</td>
      <td>0.035340</td>
      <td>0.014180</td>
      <td>0.433720</td>
      <td>0.420820</td>
      <td>0.409060</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
      <td>0.827660</td>
      <td>0.886600</td>
      <td>0.283300</td>
      <td>0.581200</td>
      <td>0.617080</td>
      <td>0.617380</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['default_next_month']==1][L].mean()
```




    %REMAINING_BAL1    0.509703
    %REMAINING_BAL2    0.513693
    %REMAINING_BAL3    0.530518
    %REMAINING_BAL4    0.560302
    %REMAINING_BAL5    0.588519
    %REMAINING_BAL6    0.601481
    dtype: float64




```python
df[df['default_next_month']==0][L].mean()
```




    %REMAINING_BAL1    0.595124
    %REMAINING_BAL2    0.610225
    %REMAINING_BAL3    0.629760
    %REMAINING_BAL4    0.663275
    %REMAINING_BAL5    0.689152
    %REMAINING_BAL6    0.704118
    dtype: float64




```python
df_rem=pd.DataFrame(df[df['default_next_month']==0][L].mean())
df_rem.columns=['no_default']
df_rem['default']=df[df['default_next_month']==1][L].mean()
df_rem=df_rem.rename_axis('Remaining credit limit').reset_index()
df_rem
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Remaining credit limit</th>
      <th>no_default</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>%REMAINING_BAL1</td>
      <td>0.595124</td>
      <td>0.509703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>%REMAINING_BAL2</td>
      <td>0.610225</td>
      <td>0.513693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>%REMAINING_BAL3</td>
      <td>0.629760</td>
      <td>0.530518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>%REMAINING_BAL4</td>
      <td>0.663275</td>
      <td>0.560302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>%REMAINING_BAL5</td>
      <td>0.689152</td>
      <td>0.588519</td>
    </tr>
    <tr>
      <th>5</th>
      <td>%REMAINING_BAL6</td>
      <td>0.704118</td>
      <td>0.601481</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax=df_rem.plot(figsize=(10,10),kind='bar',x='Remaining credit limit', stacked=False, fontsize=13)
ax.set_title("Mean remaining credit limit by default status", fontsize=18)
ax.set_ylabel("Percentage", fontsize=18)
ax.set_xticklabels(labels=df_rem['Remaining credit limit'],fontsize=10, rotation=45)
plt.savefig("rem_credit.png",transparent=True)
```


    
![png](output_68_0.png)
    



```python
print(df['%REMAINING_BAL1'].min())
print(df['%REMAINING_BAL1'].max())
print(df['%REMAINING_BAL1'].mean())
```

    -5.4553
    1.619892
    0.5762285486750798
    

### Negative Bill Amount


```python
print(df[(df['%REMAINING_BAL1']>1)].shape)
print(df[(df['%REMAINING_BAL1']>1)&(df['default_next_month']==1)].shape)
```

    (590, 30)
    (109, 30)
    


```python
df[(df['BILL_AMT1']<0)].shape
```




    (590, 30)




```python

```

### Exceeded credit limit


```python
print(df[(df['%REMAINING_BAL1']<0)].shape)
print(df[(df['%REMAINING_BAL1']<0)&(df['default_next_month']==0)].shape)
```

    (2115, 30)
    (1479, 30)
    


```python
#37 clients used more than twice of their credit limit
df[(df['%REMAINING_BAL1']<-1)]['%REMAINING_BAL1'].shape
```




    (37,)




```python
print(str(100*(2115/len(df)))+'% of all clients exceeded their credit limit in September')
```

    7.049999999999999% of all clients exceeded their credit limit in September
    

# Bill Amt & Pay Amt


```python
#L2=[]
#for i in list(range(1,7)):
    #x='%BILL_PAID'+str(i)
    #z='PAY_AMT'+str(i)
    #L2.append(x)
    #y='BILL_AMT'+str(i)
    #df[x]=(df[z])/df[y]
```

df[L2].fillna(0)

# Repayment status


```python
df['PAY_1'].value_counts()
```




     0    14737
    -1     5686
     1     3688
    -2     2759
     2     2667
     3      322
     4       76
     5       26
     8       19
     6       11
     7        9
    Name: PAY_1, dtype: int64




```python
# For consistency and since -1 and -2 have the same default probability as 0, they are regrouped under it
for i in list(range(1,7)):
    x='PAY_'+str(i)
    df.loc[df[x]==-1,x]=0
    df.loc[df[x]==-2,x]=0
```


```python
df_pay=pd.DataFrame(df['PAY_1'].loc[df['default_next_month'] == 0].value_counts())
df_pay.columns=['no_default']
df_pay['default']=df['PAY_1'].loc[df1['default_next_month'] == 1].value_counts()
df_pay['Total']=df['PAY_1'].value_counts()
df_pay['%no_default']=(df_pay['no_default']/df_pay['Total'])*100
df_pay['%default']=(df_pay['default']/df_pay['Total'])*100
df_pay=df_pay.rename_axis('PAY_1').reset_index()
df_pay=df_pay.drop(['no_default','default','Total'],axis=1)
df_pay=df_pay.fillna(0)
df_pay=df_pay.sort_values(by=['PAY_1'],ascending=False)
```


```python
df_pay
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_1</th>
      <th>%no_default</th>
      <th>%default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>42.105263</td>
      <td>57.894737</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>22.222222</td>
      <td>77.777778</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>45.454545</td>
      <td>54.545455</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>31.578947</td>
      <td>68.421053</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>24.223602</td>
      <td>75.776398</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>30.858643</td>
      <td>69.141357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>66.052061</td>
      <td>33.947939</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>86.165991</td>
      <td>13.834009</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax=df_pay.plot(figsize=(10,10),kind='barh',x='PAY_1', stacked=True, fontsize=13)
ax.set_title("Probability of default by payment delay (months)", fontsize=18)
ax.set_ylabel("Repayment delay", fontsize=18)
ax.set_xlabel("Percentage", fontsize=18)
plt.savefig("PAY_proba.png", transparent=True)
```


    
![png](output_86_0.png)
    


# Correlations 2


```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), cmap='rocket_r', annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f66c286940>




    
![png](output_88_1.png)
    


# Default prediction


```python
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, roc_curve, f1_score
```


```python
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
```


```python
from sklearn.metrics import classification_report
```

# Splitting


```python
X= df.drop('default_next_month',1)
X=X.loc[X['AGE']!=79]
y= df.loc[df['AGE']!=79]['default_next_month']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=37, stratify=y)
```


```python
X_xgb=X.copy()
xgb_col=[]
for i in X.columns.to_list():
    f ='f'+str(len(xgb_col))
    xgb_col.append(f)
```


```python
X_xgb.columns=xgb_col
X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(X_xgb,y,test_size=0.2, random_state=37, stratify=y)
```

# Informative Models

# Blind decision tree


```python
Tree = DecisionTreeClassifier(random_state=37)
```


```python
Tree.fit(X=X_train,y=y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=37,
                splitter='best')




```python
accuracy_score(y_pred=Tree.predict(X=X_test),y_true=y_test)
```




    0.7261666666666666




```python
roc_auc_score(y_score=Tree.predict_proba(X=X_test)[:, 1],y_true=y_test)
```




    0.6081756683643842




```python
feat_imp=pd.DataFrame(Tree.feature_importances_)
feat_imp=feat_imp.transpose()
feat_imp.columns=X.columns
feat_imp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>%REMAINING_BAL1</th>
      <th>%REMAINING_BAL2</th>
      <th>%REMAINING_BAL3</th>
      <th>%REMAINING_BAL4</th>
      <th>%REMAINING_BAL5</th>
      <th>%REMAINING_BAL6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038103</td>
      <td>0.009975</td>
      <td>0.019029</td>
      <td>0.013256</td>
      <td>0.059917</td>
      <td>0.157035</td>
      <td>0.030908</td>
      <td>0.010225</td>
      <td>0.002828</td>
      <td>0.007356</td>
      <td>0.005363</td>
      <td>0.039875</td>
      <td>0.026066</td>
      <td>0.030081</td>
      <td>0.028844</td>
      <td>0.031453</td>
      <td>0.035334</td>
      <td>0.037302</td>
      <td>0.036273</td>
      <td>0.047803</td>
      <td>0.039671</td>
      <td>0.034261</td>
      <td>0.040905</td>
      <td>0.043829</td>
      <td>0.040211</td>
      <td>0.031328</td>
      <td>0.036714</td>
      <td>0.033468</td>
      <td>0.032589</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,10))
g=sns.barplot(data=feat_imp)
g.set_xticklabels(rotation=30, labels=X.columns)
plt.savefig("tree_feature.png",transparent=True)
```


    
![png](output_104_0.png)
    


# Lasso


```python
lassocv = LassoCV(cv=5, alphas=[0.5,0.2,0.1,0.01,0.001], n_jobs=-1)
```


```python
lassocv.fit(X_train,y_train)
```




    LassoCV(alphas=[0.5, 0.2, 0.1, 0.01, 0.001], copy_X=True, cv=5, eps=0.001,
        fit_intercept=True, max_iter=1000, n_alphas=100, n_jobs=-1,
        normalize=False, positive=False, precompute='auto', random_state=None,
        selection='cyclic', tol=0.0001, verbose=False)




```python
lassocv.alpha_
```




    0.001




```python
lassocv.coef_
```




    array([-2.42001324e-07, -1.69869432e-02, -9.46311190e-03, -1.63681614e-02,
            1.06779430e-03,  1.58910454e-01,  1.50020679e-02,  2.63486208e-02,
            1.08218502e-02,  1.45856136e-02,  2.28895185e-02, -1.50335041e-07,
            2.34836304e-07,  8.18631842e-08, -2.39381834e-08, -2.43068754e-07,
            2.22341599e-07, -6.84554447e-07, -1.42442224e-07, -2.77403497e-08,
           -1.92873797e-07, -4.04516965e-07, -2.01023989e-07,  0.00000000e+00,
           -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
           -0.00000000e+00])




```python
X.columns[np.where(lassocv.coef_ > 0.0000000001)[0].tolist()]
```




    Index(['AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
           'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT6'],
          dtype='object')



# Informative Random Forest


```python
RF2=RandomForestClassifier(n_jobs=-1, random_state=37, class_weight='balanced', max_depth=10,max_features=5,n_estimators=150)
RF2_clf=RF2.fit(X_train, y_train)

RD_feat=pd.DataFrame(RF2.feature_importances_).transpose()
RD_feat.columns=X.columns
plt.figure(figsize=(15,10))
g=sns.barplot(data=RD_feat)
g.set_xticklabels(rotation=30, labels=X.columns)
```




    [Text(0,0,'LIMIT_BAL'),
     Text(0,0,'SEX'),
     Text(0,0,'EDUCATION'),
     Text(0,0,'MARRIAGE'),
     Text(0,0,'AGE'),
     Text(0,0,'PAY_1'),
     Text(0,0,'PAY_2'),
     Text(0,0,'PAY_3'),
     Text(0,0,'PAY_4'),
     Text(0,0,'PAY_5'),
     Text(0,0,'PAY_6'),
     Text(0,0,'BILL_AMT1'),
     Text(0,0,'BILL_AMT2'),
     Text(0,0,'BILL_AMT3'),
     Text(0,0,'BILL_AMT4'),
     Text(0,0,'BILL_AMT5'),
     Text(0,0,'BILL_AMT6'),
     Text(0,0,'PAY_AMT1'),
     Text(0,0,'PAY_AMT2'),
     Text(0,0,'PAY_AMT3'),
     Text(0,0,'PAY_AMT4'),
     Text(0,0,'PAY_AMT5'),
     Text(0,0,'PAY_AMT6'),
     Text(0,0,'%REMAINING_BAL1'),
     Text(0,0,'%REMAINING_BAL2'),
     Text(0,0,'%REMAINING_BAL3'),
     Text(0,0,'%REMAINING_BAL4'),
     Text(0,0,'%REMAINING_BAL5'),
     Text(0,0,'%REMAINING_BAL6')]




    
![png](output_112_1.png)
    


# Predictive Modelling

# No sampling - Logistic 


```python
logit=LogisticRegression(random_state=37, solver = "liblinear", max_iter=200)
parameters_logit = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20)}
grid_logit = GridSearchCV(logit, param_grid=parameters_logit, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_clf=grid_logit.fit(X_train, y_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:   28.2s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:  1.7min finished
    


```python
print('The optimal parameters for the RF are:', logit_clf.best_params_, 'with a validation ROC auc score of:', logit_clf.best_score_)
```

    The optimal parameters for the RF are: {'C': 0.08858667904100823, 'penalty': 'l1'} with a validation ROC auc score of: 0.7589429617192803
    


```python
roc_auc_score(y_score=logit_clf.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.7520789553933506



# SMOTE - Logistic 


```python
pipeline_logit = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('logit', LogisticRegression(random_state=37, solver = "liblinear"))])
parameters_logit_res = {'logit__penalty' : ['l1', 'l2'], 'logit__C' : np.logspace(-4, 4, 20)}
grid_logit_res = GridSearchCV(pipeline_logit, param_grid=parameters_logit_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_res=grid_logit_res.fit(X_train, y_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:   29.1s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:  3.7min finished
    


```python
print('The optimal parameters for the RF are:', logit_res.best_params_, 'with a validation ROC auc score of:', logit_res.best_score_)
```

    The optimal parameters for the RF are: {'logit__C': 0.012742749857031334, 'logit__penalty': 'l1'} with a validation ROC auc score of: 0.761565474741622
    


```python
roc_auc_score(y_score=logit_res.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.7550642461600585



## Borderline SMOTE - Logit


```python
pipeline_logit2 = Pipeline([('B_SMOTE',BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('logit', LogisticRegression(random_state=37, solver = "liblinear"))])
parameters_logit_res = {'logit__penalty' : ['l1', 'l2'], 'logit__C' : np.logspace(-4, 4, 20)}
grid_logit_res2 = GridSearchCV(pipeline_logit2, param_grid=parameters_logit_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_res2=grid_logit_res2.fit(X_train, y_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:   37.9s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:  3.7min
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:  4.0min finished
    

# No sampling - RF


```python
X_train.shape
```




    (23999, 29)




```python
RF=RandomForestClassifier(n_jobs=-2, random_state=37)
parameteres_RF = {'n_estimators' : list(range(100,151,10)), 'max_features' : list(range(1,11,2)), 'class_weight' : [None, 'balanced'], 'max_depth': [15, 10 , 5]}
grid_RF = GridSearchCV(RF, param_grid=parameteres_RF, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
RF_clf=grid_RF.fit(X_train, y_train)
```

    Fitting 5 folds for each of 180 candidates, totalling 900 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:   22.5s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:  3.6min
    [Parallel(n_jobs=-2)]: Done 436 tasks      | elapsed:  7.0min
    [Parallel(n_jobs=-2)]: Done 786 tasks      | elapsed: 13.0min
    [Parallel(n_jobs=-2)]: Done 900 out of 900 | elapsed: 14.2min finished
    


```python
print('The optimal parameters for the RF are:', RF_clf.best_params_, 'with a validation ROC auc score of:', RF_clf.best_score_)
```

    The optimal parameters for the RF are: {'class_weight': None, 'max_depth': 10, 'max_features': 5, 'n_estimators': 150} with a validation ROC auc score of: 0.7823416581420589
    


```python
RF_Pscore=RF_clf.predict_proba(X_test)
roc_auc_score(y_score=RF_Pscore[:, 1], y_true=y_test)
```




    0.7834404734278966



# SMOTE - RF


```python
sm = SMOTE(random_state = 37, k_neighbors=10)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
```


```python
pipeline_RF = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('RF', RandomForestClassifier(n_jobs=-1, random_state=37))])
parameteres_RF = {'RF__n_estimators' : list(range(100,151,10)), 'RF__max_features' : list(range(1,11,2)), 'RF__max_depth': [15, 10 , 5]}
grid_RF_res = GridSearchCV(pipeline_RF, param_grid=parameteres_RF, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
RF_res=grid_RF_res.fit(X_train,y_train)
```

    Fitting 5 folds for each of 90 candidates, totalling 450 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   33.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  5.7min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed: 12.0min
    [Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed: 12.3min finished
    


```python
print('The optimal parameters for the RF are:', RF_res.best_params_, 'with a validation ROC auc score of:', RF_res.best_score_)
```

    The optimal parameters for the RF are: {'RF__max_depth': 10, 'RF__max_features': 7, 'RF__n_estimators': 150} with a validation ROC auc score of: 0.7790893556504329
    


```python
roc_auc_score(y_score=RF_res.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.7760175621275744



## Borderline SMOTE - RF


```python
pipeline_RF2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('RF', RandomForestClassifier(n_jobs=-1, random_state=37))])
parameteres_RF = {'RF__n_estimators' : list(range(100,151,10)), 'RF__max_features' : list(range(1,11,2)), 'RF__max_depth': [15, 10 , 5]}
grid_RF_res2 = GridSearchCV(pipeline_RF2, param_grid=parameteres_RF, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
RF_res2=grid_RF_res2.fit(X_train,y_train)
```

    Fitting 5 folds for each of 90 candidates, totalling 450 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   40.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  5.9min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed: 12.9min
    [Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed: 13.2min finished
    

# ANN


```python
'''Sklearn is used instead of Keras for reproducibility'''
```




    'Sklearn is used instead of Keras for reproducibility'




```python
NN=MLPClassifier(random_state=37, solver='adam', batch_size=100)
parameteres_NN = {'hidden_layer_sizes' : [(100,100,100),(300,300), (300,), (200,), (500,)]}
grid_NN = GridSearchCV(NN, param_grid=parameteres_NN, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
NN_clf=grid_NN.fit(X_train, y_train)
```

    Fitting 5 folds for each of 5 candidates, totalling 25 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  25 out of  25 | elapsed:  7.9min finished
    


```python
print('The optimal parameters for the NN are:', NN_clf.best_params_, 'with a validation ROC auc score of:', NN_clf.best_score_)
```

    The optimal parameters for the NN are: {'hidden_layer_sizes': (300,)} with a validation ROC auc score of: 0.6558275599077256
    


```python
roc_auc_score(y_score=NN_clf.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.6005809319067626



# SMOTE - ANN


```python
pipeline_NN = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('NN', MLPClassifier(random_state=37, solver='adam', batch_size=100))])
parameteres_NN_res = {'NN__hidden_layer_sizes' : [(500,), (300,300), (300,), (600,)]}
grid_NN_res = GridSearchCV(pipeline_NN, param_grid=parameteres_NN_res, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
NN_clf_res=grid_NN_res.fit(X_train, y_train)
```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed: 12.5min finished
    


```python
print('The optimal parameters for the NN are:', NN_clf_res.best_params_, 'with a validation ROC auc score of:', NN_clf_res.best_score_)
```

    The optimal parameters for the NN are: {'NN__hidden_layer_sizes': (500,)} with a validation ROC auc score of: 0.6427089671447399
    


```python
roc_auc_score(y_score=NN_clf_res.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.6046128160764488



## Borderline SMOTE - ANN


```python
pipeline_NN2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('NN', MLPClassifier(random_state=37, solver='adam', batch_size=100))])
parameteres_NN_res = {'NN__hidden_layer_sizes' : [(500,), (300,300), (300,), (500,500), (600,600), (600,)], 'NN__max_iter' : [50,100,200]}
grid_NN_res2 = GridSearchCV(pipeline_NN2, param_grid=parameteres_NN_res, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
NN_clf_res2=grid_NN_res2.fit(X_train, y_train)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 22.6min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed: 167.3min finished
    

# No sampling - XGBoost


```python
XGB=XGBClassifier(n_jobs=-1, random_state=37)
parameteres_XGB = {'n_estimators' : list(range(100,151,10))}
grid_XGB = GridSearchCV(XGB, param_grid=parameteres_XGB, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
XGB_clf=grid_XGB.fit(X_train,y_train)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    D:\Anaconda3\lib\site-packages\sklearn\externals\joblib\externals\loky\process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    D:\Anaconda3\lib\site-packages\sklearn\externals\joblib\externals\loky\process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  4.3min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:  6.9min finished
    


```python
print('The optimal parameters for the NN are:', XGB_clf.best_params_, 'with a validation ROC auc score of:', XGB_clf.best_score_)
```

    The optimal parameters for the NN are: {'max_depth': 5, 'n_estimators': 100} with a validation ROC auc score of: 0.7828275934284988
    


```python
roc_auc_score(y_score=XGB_clf.predict_proba(X_test)[:, 1], y_true=y_test)
```




    0.7781055079033927



# SMOTE - XGBoost


```python
pipeline_XGB = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('XGB', XGBClassifier(n_jobs=-1, random_state=37))])
parameteres_XGB_res = {'XGB__n_estimators' : list(range(100,151,10))}
grid_XGB_res = GridSearchCV(pipeline_XGB, param_grid=parameteres_XGB_res, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
XGB_clf_res=grid_XGB_res.fit(X_xgb_train,y_xgb_train)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  5.1min
    [Parallel(n_jobs=-1)]: Done 120 out of 120 | elapsed:  9.9min finished
    


```python
print('The optimal parameters for the NN are:', XGB_clf_res.best_params_, 'with a validation ROC auc score of:', XGB_clf_res.best_score_)
```

    The optimal parameters for the NN are: {'XGB__max_depth': 3, 'XGB__n_estimators': 150} with a validation ROC auc score of: 0.7803199691575697
    


```python
roc_auc_score(y_score=XGB_clf_res.predict_proba(X_xgb_test)[:, 1], y_true=y_test)
```




    0.7774543300665321



## Borderline SMOTE - XGBoost


```python
pipeline_XGB2 = Pipeline([('SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('XGB', XGBClassifier(n_jobs=-1, random_state=37))])
parameteres_XGB_res = {'XGB__n_estimators' : list(range(100,151,10))}
grid_XGB_res2 = GridSearchCV(pipeline_XGB2, param_grid=parameteres_XGB_res, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
XGB_clf_res2=grid_XGB_res2.fit(X_xgb_train,y_xgb_train)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  4.7min
    [Parallel(n_jobs=-1)]: Done 120 out of 120 | elapsed:  9.6min finished
    

# REDUCED MODEL


```python
X_red=X[['PAY_1','PAY_AMT1','PAY_AMT2','PAY_AMT3','%REMAINING_BAL1','%REMAINING_BAL2','%REMAINING_BAL3']]

X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_red,y,test_size=0.2, random_state=37, stratify=y)

X_xgb_red=X_red.copy()
xgb_col2=[]
for i in X_red.columns.to_list():
    f ='f'+str(len(xgb_col2))
    xgb_col2.append(f)

X_xgb_red.columns=xgb_col2
X_xgb_red_train, X_xgb_red_test, y_xgb_red_train, y_xgb_red_test = train_test_split(X_xgb_red,y,test_size=0.2, random_state=37, stratify=y)
```

# No Sampling - Reduced - logit


```python
logit=LogisticRegression(random_state=37, solver = "liblinear", max_iter=200)
parameters_logit = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20)}
grid_logit = GridSearchCV(logit, param_grid=parameters_logit, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_red=grid_logit.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:   17.4s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:   26.9s
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:   27.7s finished
    


```python
print('The optimal parameters for the RF are:', logit_red.best_params_, 'with a validation ROC auc score of:', logit_red.best_score_)
```

    The optimal parameters for the RF are: {'C': 0.0018329807108324356, 'penalty': 'l1'} with a validation ROC auc score of: 0.7473242005607136
    


```python
roc_auc_score(y_score=logit_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7442825602222584



# SMOTE - Reduced - logit


```python
pipeline_logit = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('logit', LogisticRegression(random_state=37, solver = "liblinear"))])
parameters_logit_res = {'logit__penalty' : ['l1', 'l2'], 'logit__C' : np.logspace(-4, 4, 20)}
grid_logit_res = GridSearchCV(pipeline_logit, param_grid=parameters_logit_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_res_red=grid_logit_res.fit(X_red_train, y_red_train)
```

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:    3.4s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:   20.6s
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:   22.3s finished
    


```python
print('The optimal parameters for the RF are:', logit_res_red.best_params_, 'with a validation ROC auc score of:', logit_res_red.best_score_)
```

    The optimal parameters for the RF are: {'logit__C': 0.0001, 'logit__penalty': 'l2'} with a validation ROC auc score of: 0.7456832371121671
    


```python
roc_auc_score(y_score=logit_res_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7399532274344223



## Borderline SMOTE - Reduced - logit


```python
pipeline_logit2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('logit', LogisticRegression(random_state=37, solver = "liblinear"))])
parameters_logit_res = {'logit__penalty' : ['l1', 'l2'], 'logit__C' : np.logspace(-4, 4, 20)}
grid_logit_res2 = GridSearchCV(pipeline_logit2, param_grid=parameters_logit_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
logit_res_red2=grid_logit_res2.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  36 tasks      | elapsed:    4.3s
    [Parallel(n_jobs=-2)]: Done 186 tasks      | elapsed:   26.1s
    [Parallel(n_jobs=-2)]: Done 200 out of 200 | elapsed:   27.9s finished
    

# No Sampling - Reduced - RF


```python
X_red_train.shape
```




    (23999, 7)




```python
list(range(1,8,1))
```




    [1, 2, 3, 4, 5, 6, 7]




```python
RF_red=RandomForestClassifier(n_jobs=-1, random_state=37)
parameteres_RF_red = {'n_estimators' : list(range(100,151,10)), 'max_features' : list(range(1,8,1)), 'class_weight' : [None, 'balanced'], 'max_depth': [15, 10 , 5]}
grid_RF_red = GridSearchCV(RF_red, param_grid=parameteres_RF_red, cv=5, verbose=True, n_jobs=-1, scoring='roc_auc')
```


```python
RF_clf_red=grid_RF_red.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 252 candidates, totalling 1260 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   22.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  3.9min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  8.2min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed: 13.0min
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed: 19.8min
    [Parallel(n_jobs=-1)]: Done 1260 out of 1260 | elapsed: 20.2min finished
    


```python
print('The optimal parameters for the RF are:', RF_clf_red.best_params_, 'with a validation ROC auc score of:', RF_clf_red.best_score_)
```

    The optimal parameters for the RF are: {'class_weight': None, 'max_depth': 10, 'max_features': 2, 'n_estimators': 130} with a validation ROC auc score of: 0.7737793197182686
    


```python
roc_auc_score(y_score=RF_clf_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7642910394027096



# SMOTE - Reduced - RF


```python
pipeline_RF_red = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('RF', RandomForestClassifier(n_jobs=-1, random_state=37))])
parameteres_RF_res_red = {'RF__n_estimators' : list(range(100,151,10)), 'RF__max_features' : list(range(1,8,1)), 'RF__max_depth': [15, 10 , 5]}
grid_RF_res_red = GridSearchCV(pipeline_RF_red, param_grid=parameteres_RF_res_red, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
RF_res_red=grid_RF_res_red.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 126 candidates, totalling 630 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   34.8s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  5.5min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed: 11.9min
    [Parallel(n_jobs=-1)]: Done 630 out of 630 | elapsed: 15.1min finished
    


```python
print('The optimal parameters for the RF are:', RF_res_red.best_params_, 'with a validation ROC auc score of:', RF_res_red.best_score_)
```

    The optimal parameters for the RF are: {'RF__max_depth': 5, 'RF__max_features': 3, 'RF__n_estimators': 130} with a validation ROC auc score of: 0.764012233382162
    


```python
roc_auc_score(y_score=RF_res_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7539179925532218



## Borderline SMOTE - Reduced - RF


```python
pipeline_RF_red2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('RF', RandomForestClassifier(n_jobs=-1, random_state=37))])
parameteres_RF_res_red = {'RF__n_estimators' : list(range(100,151,10)), 'RF__max_features' : list(range(1,8,1)), 'RF__max_depth': [15, 10 , 5]}
grid_RF_res_red2 = GridSearchCV(pipeline_RF_red2, param_grid=parameteres_RF_res_red, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
RF_res_red2=grid_RF_res_red2.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 126 candidates, totalling 630 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   33.0s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  5.3min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed: 11.5min
    [Parallel(n_jobs=-1)]: Done 630 out of 630 | elapsed: 14.7min finished
    

# No Sampling - Reduced - ANN


```python
NN_red=MLPClassifier(random_state=37, solver='adam', batch_size=100)
parameteres_NN_red = {'hidden_layer_sizes' : [(500,),(300,300), (300,), (500,500), (600,600), (600,)]}
grid_NN_red = GridSearchCV(NN_red, param_grid=parameteres_NN_red, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
NN_clf_red=grid_NN_red.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  30 out of  30 | elapsed: 29.4min finished
    


```python
print('The optimal parameters for the NN are:', NN_clf_red.best_params_, 'with a validation ROC auc score of:', NN_clf_red.best_score_)
```

    The optimal parameters for the NN are: {'hidden_layer_sizes': (500,)} with a validation ROC auc score of: 0.7088098208045374
    


```python
roc_auc_score(y_score=NN_clf_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7289452096258856



# SMOTE - Reduced - ANN


```python
pipeline_NN_red = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('NN', MLPClassifier(random_state=37,max_iter=400, solver='adam', batch_size=100))])
parameteres_NN_res_red = {'NN__hidden_layer_sizes' : [(500,),(300,300), (300,), (500,500), (600,600), (600,)]}
grid_NN_res_red = GridSearchCV(pipeline_NN_red, param_grid=parameteres_NN_res_red, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
NN_res_red=grid_NN_res_red.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 33.1min finished
    


```python
print('The optimal parameters for the NN are:', NN_res_red.best_params_, 'with a validation ROC auc score of:', NN_res_red.best_score_)
```

    The optimal parameters for the NN are: {'NN__hidden_layer_sizes': (500,)} with a validation ROC auc score of: 0.7399838961051123
    


```python
roc_auc_score(y_score=NN_res_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7404461261611099



## Borderline SMOTE - Reduced - ANN


```python
pipeline_NN_red2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('NN', MLPClassifier(random_state=37,max_iter=400, solver='adam', batch_size=100))])
parameteres_NN_res_red = {'NN__hidden_layer_sizes' : [(500,),(300,300), (300,), (500,500), (600,600), (600,)], 'NN__max_iter' : [50,100,200]}
grid_NN_res_red2 = GridSearchCV(pipeline_NN_red2, param_grid=parameteres_NN_res_red, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
NN_res_red2=grid_NN_res_red2.fit(X_red_train, y_red_train)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 15.8min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed: 98.0min finished
    


```python
roc_auc_score(y_score=NN_res_red2.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7430813161145874



# Reduced - XGB


```python
XGB=XGBClassifier(n_jobs=-1, random_state=37)
parameteres_XGB = {'n_estimators' : list(range(100,151,10))}
grid_XGB = GridSearchCV(XGB, param_grid=parameteres_XGB, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
XGB_clf_red=grid_XGB.fit(X_red_train,y_red_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   10.9s finished
    


```python
print('The optimal parameters for the NN are:', XGB_clf_red.best_params_, 'with a validation ROC auc score of:', XGB_clf_red.best_score_)
```

    The optimal parameters for the NN are: {'n_estimators': 110} with a validation ROC auc score of: 0.7724900668022684
    


```python
roc_auc_score(y_score=XGB_clf_red.predict_proba(X_red_test)[:, 1], y_true=y_test)
```




    0.7640473718169007



# SMOTE - Reduced - XGB


```python
pipeline_XGB = Pipeline([('SMOTE', SMOTE(random_state = 37, k_neighbors=10)), ('XGB', XGBClassifier(n_jobs=-1, random_state=37))])
parameteres_XGB_res = {'XGB__n_estimators' : list(range(100,151,10))}
grid_XGB_res = GridSearchCV(pipeline_XGB, param_grid=parameteres_XGB_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
XGB_clf_res_red=grid_XGB_res.fit(X_xgb_red_train,y_xgb_red_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   17.1s finished
    


```python
print('The optimal parameters for the NN are:', XGB_clf_res_red.best_params_, 'with a validation ROC auc score of:', XGB_clf_res_red.best_score_)
```

    The optimal parameters for the NN are: {'XGB__n_estimators': 100} with a validation ROC auc score of: 0.7615271083033442
    


```python
roc_auc_score(y_score=XGB_clf_res_red.predict_proba(X_xgb_red_test)[:, 1], y_true=y_test)
```




    0.7530571735108338



## Borderline SMOTE - Reduced - XGBoost


```python
pipeline_XGB2 = Pipeline([('B_SMOTE', BorderlineSMOTE(random_state = 37, k_neighbors=5)), ('XGB', XGBClassifier(n_jobs=-1, random_state=37))])
parameteres_XGB_res = {'XGB__n_estimators' : list(range(100,151,10))}
grid_XGB_res2 = GridSearchCV(pipeline_XGB2, param_grid=parameteres_XGB_res, cv=5, verbose=True, n_jobs=-2, scoring='roc_auc')
```


```python
XGB_clf_res_red2=grid_XGB_res2.fit(X_xgb_red_train,y_xgb_red_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   18.1s finished
    


```python
roc_auc_score(y_score=XGB_clf_res_red2.predict_proba(X_xgb_red_test)[:, 1], y_true=y_test)
```




    0.7557403390478838



# Results


```python
CLF_list=[logit_clf,logit_res,logit_red,logit_res_red,RF_clf,RF_res,RF_clf_red,RF_res_red,NN_clf,NN_clf_res,NN_clf_red,NN_res_red,XGB_clf,XGB_clf_res,XGB_clf_red,XGB_clf_res_red,logit_res2,logit_res_red2, RF_res2, RF_res_red2, NN_clf_res2, NN_res_red2, XGB_clf_res2, XGB_clf_res_red2]
CLF_list_str=['logit_clf','logit_res','logit_red','logit_res_red','RF_clf','RF_res','RF_red','RF_res_red','NN_clf','NN_clf_res','NN_red','NN_res_red','XGB_clf','XGB_clf_res','XGB_clf_red','XGB_clf_res_red','logit_res2','logit_res_red2', 'RF_res2', 'RF_res_red2', 'NN_clf_res2', 'NN_res_red2', 'XGB_clf_res2', 'XGB_clf_res_red2']
legend=['base line','logit_clf','logit_res','logit_red','logit_res_red','RF_clf','RF_res','RF_red','RF_res_red','NN_clf','NN_clf_res','NN_red','NN_res_red','XGB_clf','XGB_clf_res','XGB_clf_red','XGB_clf_res_red','logit_res2','logit_res_red2', 'RF_res2', 'RF_res_red2', 'NN_clf_res2', 'NN_res_red2', 'XGB_clf_res2', 'XGB_clf_res_red2']
```

# ROC curve


```python
plt.figure(figsize=(15,10))
plt.axis([0, 1, 0, 1])
plt.title('ROC Curves')
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        fpr,tpr,threshold = roc_curve(y_test, CLF_list[i].predict_proba(X_xgb_red_test)[:, 1])
        plt.plot(fpr, tpr, label=threshold)
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        fpr,tpr,threshold = roc_curve(y_test, CLF_list[i].predict_proba(X_xgb_test)[:, 1])
        plt.plot(fpr, tpr, label=threshold)
    elif 'red' in CLF_list_str[i]:
        fpr,tpr,threshold = roc_curve(y_test, CLF_list[i].predict_proba(X_red_test)[:, 1])
        plt.plot(fpr, tpr, label=threshold)
    else:
        fpr,tpr,threshold = roc_curve(y_test, CLF_list[i].predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=threshold)
plt.plot([0, 1], [0, 1], '#0C8EE0', linestyle='--')
plt.legend(legend)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```




    Text(0,0.5,'True Positive Rate')




    
![png](output_230_1.png)
    



```python
plt.figure(figsize=(15,10))
plt.axis([0, 1, 0, 1])
plt.title('ROC Curve of the Random Forest (Borderline SMOTE)')
fpr,tpr,threshold = roc_curve(y_test, RF_res2.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, RF_clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, NN_res_red.predict_proba(X_red_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
plt.plot([0, 1], [0, 1], '#0C8EE0', linestyle='--')
plt.legend(['RF (Borderline SMOTE)','RF','ANN (Reduced)(SMOTE)','base line'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("RF_res2.png",transparent=True)
```


    
![png](output_231_0.png)
    



```python
plt.figure(figsize=(15,10))
plt.axis([0, 1, 0, 1])
plt.title('ROC Curve of the Random Forest (Borderline SMOTE)')
fpr,tpr,threshold = roc_curve(y_test, XGB_clf_red.predict_proba(X_red_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, RF_res2.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, RF_clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, NN_res_red.predict_proba(X_red_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
fpr,tpr,threshold = roc_curve(y_test, NN_clf_res2.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=threshold)
plt.plot([0, 1], [0, 1], '#0C8EE0', linestyle='--')
plt.legend(['XGBoost (Reduced)','RF (Borderline SMOTE1)','RF','ANN (Reduced)(SMOTE)','ANN (Borderline SMOTE1)','base line'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("model_time.png",transparent=True)
```


    
![png](output_232_0.png)
    


# AUC Score


```python
AUC_dict={}
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = roc_auc_score(y_true=y_test,y_score=CLF_list[i].predict_proba(X_xgb_red_test)[:,1])
        AUC_dict[CLF_list_str[i]]=r
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = roc_auc_score(y_true=y_test,y_score=CLF_list[i].predict_proba(X_xgb_test)[:,1])
        AUC_dict[CLF_list_str[i]]=r
    elif 'red' in CLF_list_str[i]:
        r = roc_auc_score(y_true=y_test,y_score=CLF_list[i].predict_proba(X_red_test)[:,1])
        AUC_dict[CLF_list_str[i]]=r
    else:
        r = roc_auc_score(y_true=y_test,y_score=CLF_list[i].predict_proba(X_test)[:,1])
        AUC_dict[CLF_list_str[i]]=r
```


```python
print('AUC scores:')
for key, value in sorted(AUC_dict.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
```

    AUC scores:
    RF_clf: 0.7879874299133165
    RF_res2: 0.7823572734451839
    RF_res: 0.7810862833210586
    XGB_clf: 0.7803363322239013
    XGB_clf_res: 0.7780077828491241
    XGB_clf_res2: 0.7760953712673182
    RF_red: 0.7642910394027096
    XGB_clf_red: 0.7640473718169007
    XGB_clf_res_red2: 0.7557403390478838
    RF_res_red2: 0.7551827740724143
    RF_res_red: 0.7539179925532218
    logit_res: 0.7532012615240173
    XGB_clf_res_red: 0.7530571735108338
    logit_res2: 0.7513175546611223
    logit_clf: 0.7503538501655601
    logit_red: 0.7442825602222584
    NN_res_red2: 0.7430813161145874
    logit_res_red2: 0.7418429816397845
    NN_res_red: 0.7404461261611099
    logit_res_red: 0.7399532274344223
    NN_red: 0.7289452096258856
    NN_clf_res2: 0.6152046960920138
    NN_clf_res: 0.5981763472793651
    NN_clf: 0.5962661127408475
    

# F1-score


```python
F1_dict={}
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = f1_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_red_test))
        F1_dict[CLF_list_str[i]]=r
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = f1_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_test))
        F1_dict[CLF_list_str[i]]=r
    elif 'red' in CLF_list_str[i]:
        r = f1_score(y_true=y_test,y_pred=CLF_list[i].predict(X_red_test))
        F1_dict[CLF_list_str[i]]=r
    else:
        r = f1_score(y_true=y_test,y_pred=CLF_list[i].predict(X_test))
        F1_dict[CLF_list_str[i]]=r
```


```python
print('F1 scores:')
for key, value in sorted(F1_dict.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
```

    F1 scores:
    logit_res: 0.5355618776671408
    logit_res2: 0.5303945965161748
    XGB_clf_res_red: 0.529304696032035
    RF_res_red2: 0.5255311218784943
    XGB_clf_res_red2: 0.5251594613749114
    RF_res_red: 0.5230648944487881
    logit_res_red: 0.5209444021325209
    logit_res_red2: 0.5190207156308851
    RF_res: 0.5137946630483944
    RF_res2: 0.5115228197017623
    NN_res_red2: 0.509979353062629
    NN_res_red: 0.5033512064343163
    XGB_clf_res: 0.49122807017543857
    RF_clf: 0.4834834834834835
    XGB_clf_res2: 0.4833659491193738
    XGB_clf: 0.4733963202386872
    XGB_clf_red: 0.4580384226491405
    RF_red: 0.45786802030456847
    logit_clf: 0.44536082474226807
    logit_red: 0.4067063277447269
    NN_clf_res: 0.38944422231107556
    NN_clf: 0.3827392120075047
    NN_clf_res2: 0.3795124961431657
    NN_red: 0.08404558404558404
    

# F_Beta-score (Beta=3)


```python
FB_dict={}
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = fbeta_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_red_test), beta=3)
        FB_dict[CLF_list_str[i]]=r
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = fbeta_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_test), beta=3)
        FB_dict[CLF_list_str[i]]=r
    elif 'red' in CLF_list_str[i]:
        r = fbeta_score(y_true=y_test,y_pred=CLF_list[i].predict(X_red_test), beta=3)
        FB_dict[CLF_list_str[i]]=r
    else:
        r = fbeta_score(y_true=y_test,y_pred=CLF_list[i].predict(X_test), beta=3)
        FB_dict[CLF_list_str[i]]=r
```


```python
print('F_Beta=3 scores:')
for key, value in sorted(FB_dict.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
```

    F_Beta=3 scores:
    NN_clf_res2: 0.7193823839045502
    NN_clf_res: 0.6236393904469202
    logit_res: 0.5607685433422698
    logit_res2: 0.5555141857174771
    NN_res_red: 0.5522058823529412
    XGB_clf_res_red2: 0.5514213424616757
    NN_clf: 0.5483870967741935
    NN_res_red2: 0.5479958586007987
    XGB_clf_res_red: 0.5440395120856095
    RF_res_red2: 0.5301150462440786
    logit_res_red2: 0.5191771531911687
    logit_res_red: 0.5165382872677843
    RF_res_red: 0.5078184302413845
    RF_res: 0.4428159351368207
    RF_res2: 0.44118793358796476
    XGB_clf_res: 0.3978528575939375
    XGB_clf_res2: 0.39020537124802523
    RF_clf: 0.38290788013318544
    XGB_clf: 0.3769699849528787
    XGB_clf_red: 0.35969509290138163
    RF_red: 0.35833465755601457
    logit_clf: 0.34405861739407456
    logit_red: 0.3016446048937024
    NN_red: 0.049084858569051586
    

# Recall


```python
recall_dict={}
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = recall_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_red_test))
        recall_dict[CLF_list_str[i]]=r
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = recall_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_test))
        recall_dict[CLF_list_str[i]]=r
    elif 'red' in CLF_list_str[i]:
        r = recall_score(y_true=y_test,y_pred=CLF_list[i].predict(X_red_test))
        recall_dict[CLF_list_str[i]]=r
    else:
        r = recall_score(y_true=y_test,y_pred=CLF_list[i].predict(X_test))
        recall_dict[CLF_list_str[i]]=r
```


```python
for key, value in sorted(recall_dict.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
```

    NN_clf_res2: 0.9269027882441597
    NN_clf_res: 0.7339864355689525
    NN_clf: 0.6149208741522231
    logit_res: 0.5674453654860587
    NN_res_red: 0.5659382064807837
    logit_res2: 0.5621703089675961
    NN_res_red2: 0.5584024114544084
    XGB_clf_res_red2: 0.5584024114544084
    XGB_clf_res_red: 0.5478522984174831
    RF_res_red2: 0.5312735493594575
    logit_res_red2: 0.519216277317257
    logit_res_red: 0.5154483798040693
    RF_res_red: 0.5041446872645065
    RF_res: 0.42803315749811605
    RF_res2: 0.426525998492841
    XGB_clf_res: 0.37980406932931426
    XGB_clf_res2: 0.37226827430293896
    RF_clf: 0.36397889977392617
    XGB_clf: 0.35870384325546345
    XGB_clf_red: 0.3413715146948003
    RF_red: 0.3398643556895252
    logit_clf: 0.3255463451394122
    logit_red: 0.28334589299171065
    NN_red: 0.044461190655614165
    

# Precision


```python
precision_dict={}
for i in range(len(CLF_list)):
    if 'red' in CLF_list_str[i] and 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = precision_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_red_test))
        precision_dict[CLF_list_str[i]]=r
    elif 'XGB' in CLF_list_str[i] and 'res' in CLF_list_str[i]:
        r = precision_score(y_true=y_test,y_pred=CLF_list[i].predict(X_xgb_test))
        precision_dict[CLF_list_str[i]]=r
    elif 'red' in CLF_list_str[i]:
        r = precision_score(y_true=y_test,y_pred=CLF_list[i].predict(X_red_test))
        precision_dict[CLF_list_str[i]]=r
    else:
        r = precision_score(y_true=y_test,y_pred=CLF_list[i].predict(X_test))
        precision_dict[CLF_list_str[i]]=r
```


```python
for key, value in sorted(precision_dict.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
```

    NN_red: 0.7662337662337663
    logit_red: 0.7203065134099617
    RF_clf: 0.7198211624441133
    logit_clf: 0.7047308319738989
    RF_red: 0.7013996889580093
    XGB_clf: 0.695906432748538
    XGB_clf_red: 0.695852534562212
    XGB_clf_res: 0.6951724137931035
    XGB_clf_res2: 0.6889818688981869
    RF_res: 0.6425339366515838
    RF_res2: 0.6388261851015802
    RF_res_red: 0.5434606011372868
    logit_res_red: 0.5265588914549654
    RF_res_red2: 0.5199115044247787
    logit_res_red2: 0.5188253012048193
    XGB_clf_res_red: 0.5119718309859155
    logit_res: 0.5070707070707071
    logit_res2: 0.5020188425302826
    XGB_clf_res_red2: 0.4956521739130435
    NN_res_red2: 0.46928435718809375
    NN_res_red: 0.4532287266143633
    NN_clf: 0.27783452502553624
    NN_clf_res: 0.26503401360544215
    NN_clf_res2: 0.23860329776915615
    
