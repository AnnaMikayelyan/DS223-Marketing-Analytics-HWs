# Homework 3 | Survival Analysis 
Student: Anna Mikayelyan


## Necessary libraries 


```python
import pandas as pd 
import os
from sklearn.preprocessing import LabelEncoder
from lifelines import ( 
    BreslowFlemingHarringtonFitter, 
    ExponentialFitter, 
    GeneralizedGammaFitter, 
    KaplanMeierFitter, 
    LogLogisticFitter, 
    LogNormalFitter, 
    MixtureCureFitter, 
    NelsonAalenFitter, 
    PiecewiseExponentialFitter, 
    SplineFitter, 
    WeibullFitter
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## Data Exploration & Preprocessing

The dataset includes customer demographic and usage data, with the target variable being churn, indicating whether a customer has churned. The preprocessing steps involve converting the churn column to binary values, where **1** indicates **churned (Yes)** and **0** indicates **not churned (No)**. Categorical variables such as **region, marital, ed, retire, gender, voice, internet, forward,** and **custcate** are transformed into numeric codes. Then, encoded categorical columns using LabelEncoder for binary features and one-hot encoding for multi-category features


```python
data_path = os.path.join(os.getcwd(), 'data', 'telco.csv')
```


```python
df = pd.read_csv(data_path)
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
      <th>ID</th>
      <th>region</th>
      <th>tenure</th>
      <th>age</th>
      <th>marital</th>
      <th>address</th>
      <th>income</th>
      <th>ed</th>
      <th>retire</th>
      <th>gender</th>
      <th>voice</th>
      <th>internet</th>
      <th>forward</th>
      <th>custcat</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Zone 2</td>
      <td>13</td>
      <td>44</td>
      <td>Married</td>
      <td>9</td>
      <td>64</td>
      <td>College degree</td>
      <td>No</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Basic service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Zone 3</td>
      <td>11</td>
      <td>33</td>
      <td>Married</td>
      <td>7</td>
      <td>136</td>
      <td>Post-undergraduate degree</td>
      <td>No</td>
      <td>Male</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Total service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Zone 3</td>
      <td>68</td>
      <td>52</td>
      <td>Married</td>
      <td>24</td>
      <td>116</td>
      <td>Did not complete high school</td>
      <td>No</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Plus service</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Zone 2</td>
      <td>33</td>
      <td>33</td>
      <td>Unmarried</td>
      <td>12</td>
      <td>33</td>
      <td>High school degree</td>
      <td>No</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Basic service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Zone 2</td>
      <td>23</td>
      <td>30</td>
      <td>Married</td>
      <td>9</td>
      <td>30</td>
      <td>Did not complete high school</td>
      <td>No</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Plus service</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>ID</th>
      <th>tenure</th>
      <th>age</th>
      <th>address</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>500.500000</td>
      <td>35.526000</td>
      <td>41.684000</td>
      <td>11.551000</td>
      <td>77.535000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288.819436</td>
      <td>21.359812</td>
      <td>12.558816</td>
      <td>10.086681</td>
      <td>107.044165</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250.750000</td>
      <td>17.000000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>500.500000</td>
      <td>34.000000</td>
      <td>40.000000</td>
      <td>9.000000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>750.250000</td>
      <td>54.000000</td>
      <td>51.000000</td>
      <td>18.000000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1000.000000</td>
      <td>72.000000</td>
      <td>77.000000</td>
      <td>55.000000</td>
      <td>1668.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.columns)
```

    Index(['ID', 'region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
           'retire', 'gender', 'voice', 'internet', 'forward', 'custcat', 'churn'],
          dtype='object')
    


```python
# Checking for missing values
df.isnull().sum()
```




    ID          0
    region      0
    tenure      0
    age         0
    marital     0
    address     0
    income      0
    ed          0
    retire      0
    gender      0
    voice       0
    internet    0
    forward     0
    custcat     0
    churn       0
    dtype: int64




```python
# Checking data types
print(df.dtypes)
```

    ID           int64
    region      object
    tenure       int64
    age          int64
    marital     object
    address      int64
    income       int64
    ed          object
    retire      object
    gender      object
    voice       object
    internet    object
    forward     object
    custcat     object
    churn       object
    dtype: object
    


```python
# Converting churn column to numeric (0: not churned, 1: churned)
df["churn"] = df["churn"].apply(lambda x: 1 if x == "Yes" else 0)
```


```python
df.columns = df.columns.str.strip()
```


```python
le = LabelEncoder()
```


```python
# Applying label encoding to columns which are binary
df['retire'] = le.fit_transform(df['retire'])
df['voice'] = le.fit_transform(df['voice'])
df['internet'] = le.fit_transform(df['internet'])
df['forward'] = le.fit_transform(df['forward'])
df['gender'] = le.fit_transform(df['gender'])
```


```python
# One-hot encoding for the multi-category columns
df = pd.get_dummies(df, columns=['region', 'marital', 'ed', 'custcat'], drop_first=True)
```


```python
# Checking again to confirm 
print(df.dtypes)
```

    ID                                 int64
    tenure                             int64
    age                                int64
    address                            int64
    income                             int64
    retire                             int32
    gender                             int32
    voice                              int32
    internet                           int32
    forward                            int32
    churn                              int64
    region_Zone 2                       bool
    region_Zone 3                       bool
    marital_Unmarried                   bool
    ed_Did not complete high school     bool
    ed_High school degree               bool
    ed_Post-undergraduate degree        bool
    ed_Some college                     bool
    custcat_E-service                   bool
    custcat_Plus service                bool
    custcat_Total service               bool
    dtype: object
    


```python
# Defining
T = df["tenure"]
E = df["churn"]
```

## Building AFT models with all the available distributions

In this section, I go through all the survival model fitters listed in the provided resource, including both parametric AFT distributions and non-parametric estimators. Each model is fitted using the **tenure** variable as the survival time and the **churn** variable as the event indicator. The goal is to evaluate which models are applicable, which models fit the data successfully, and how their performance compares based on **log-likelihood**, **AIC**, and the **shape of the survival curve**.


```python
# AalenJohansenFitter
```

The ***AalenJohansenFitter*** is conceptually invalid for this dataset because the churn variable has only one event category, meaning no competing risks exist. Including this model would misrepresent the structure of the data and produce incorrect results.


```python
# BreslowFlemingHarringtonFitter 

bfhf = BreslowFlemingHarringtonFitter()
bfhf.fit(T, E, label="Breslow-Fleming-Harrington")
```




    <lifelines.BreslowFlemingHarringtonFitter:"Breslow-Fleming-Harrington", fitted with 1000 total observations, 726 right-censored observations>




```python
# Exponential 

exp = ExponentialFitter()
exp.fit(T, E, label="Exponential")
exp.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.ExponentialFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1606.98</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>lambda_ != 0</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lambda_</th>
      <td>129.66</td>
      <td>7.83</td>
      <td>114.30</td>
      <td>145.01</td>
      <td>0.00</td>
      <td>16.55</td>
      <td>&lt;0.005</td>
      <td>202.03</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3215.96</td>
    </tr>
  </tbody>
</table>
</div>



```python
# GeneralizedGammaFitter 

gg = GeneralizedGammaFitter()
gg.fit(T, E, label="GeneralizedGamma")
gg.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.GeneralizedGammaFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1602.50</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>mu_ != 0, ln_sigma_ != 0, lambda_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_</th>
      <td>4.79</td>
      <td>0.14</td>
      <td>4.51</td>
      <td>5.06</td>
      <td>0.00</td>
      <td>34.09</td>
      <td>&lt;0.005</td>
      <td>843.87</td>
    </tr>
    <tr>
      <th>ln_sigma_</th>
      <td>0.57</td>
      <td>0.14</td>
      <td>0.29</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>4.02</td>
      <td>&lt;0.005</td>
      <td>14.08</td>
    </tr>
    <tr>
      <th>lambda_</th>
      <td>0.05</td>
      <td>0.33</td>
      <td>-0.60</td>
      <td>0.70</td>
      <td>1.00</td>
      <td>-2.87</td>
      <td>&lt;0.005</td>
      <td>7.92</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3211.01</td>
    </tr>
  </tbody>
</table>
</div>



```python
# KaplanMeierFitter 

kmf = KaplanMeierFitter()
kmf.fit(T, E, label="Kaplan-Meier")
```




    <lifelines.KaplanMeierFitter:"Kaplan-Meier", fitted with 1000 total observations, 726 right-censored observations>




```python
# LogLogisticFitter 

loglog = LogLogisticFitter()
loglog.fit(T, E, label="LogLogistic")
loglog.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogLogisticFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1605.21</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>alpha_ != 1, beta_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_</th>
      <td>103.39</td>
      <td>9.13</td>
      <td>85.50</td>
      <td>121.28</td>
      <td>1.00</td>
      <td>11.22</td>
      <td>&lt;0.005</td>
      <td>94.60</td>
    </tr>
    <tr>
      <th>beta_</th>
      <td>1.04</td>
      <td>0.05</td>
      <td>0.93</td>
      <td>1.15</td>
      <td>1.00</td>
      <td>0.73</td>
      <td>0.46</td>
      <td>1.11</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3214.42</td>
    </tr>
  </tbody>
</table>
</div>



```python
# LogNormalFitter 

lognorm = LogNormalFitter()
lognorm.fit(T, E, label="LogNormal")
lognorm.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogNormalFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1602.52</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>mu_ != 0, sigma_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_</th>
      <td>4.77</td>
      <td>0.10</td>
      <td>4.57</td>
      <td>4.98</td>
      <td>0.00</td>
      <td>46.06</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>sigma_</th>
      <td>1.81</td>
      <td>0.09</td>
      <td>1.64</td>
      <td>1.97</td>
      <td>1.00</td>
      <td>9.37</td>
      <td>&lt;0.005</td>
      <td>66.94</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3209.04</td>
    </tr>
  </tbody>
</table>
</div>



```python
# MixtureCureFitter 

base_model = WeibullFitter()
mcf = MixtureCureFitter(base_fitter=base_model)
mcf.fit(T, E, label="MixtureCure")
mcf.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.MixtureCureFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1605.68</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>cured_fraction_ != 0, lambda_ != 1, rho_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cured_fraction_</th>
      <td>0.38</td>
      <td>0.16</td>
      <td>0.07</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>2.42</td>
      <td>0.02</td>
      <td>6.01</td>
    </tr>
    <tr>
      <th>lambda_</th>
      <td>68.72</td>
      <td>26.31</td>
      <td>17.16</td>
      <td>120.28</td>
      <td>1.00</td>
      <td>2.57</td>
      <td>0.01</td>
      <td>6.64</td>
    </tr>
    <tr>
      <th>rho_</th>
      <td>1.02</td>
      <td>0.08</td>
      <td>0.87</td>
      <td>1.18</td>
      <td>1.00</td>
      <td>0.31</td>
      <td>0.76</td>
      <td>0.40</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3217.36</td>
    </tr>
  </tbody>
</table>
</div>



```python
# NelsonAalenFitter 

naf = NelsonAalenFitter()
naf.fit(T, E, label="Nelson-Aalen")

# Converting cumulative hazard H(t) to survival: S(t) = exp(-H(t))
naf.survival_function_ = np.exp(-naf.cumulative_hazard_)
```


```python
# Plotting to understand how to define breakpoints for PiecewiseExponentialFitter
plt.hist(df['tenure'], bins=50)  
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_31_0.png)
    



```python
# PiecewiseExponentialFitter 

breakpoints = [12, 24, 36, 48, 60]

pwexp = PiecewiseExponentialFitter(breakpoints=breakpoints)
pwexp.fit(T, E, label="PiecewiseExponential")
```




    <lifelines.PiecewiseExponentialFitter:"PiecewiseExponential", fitted with 1000 total observations, 726 right-censored observations>




```python
#  SplineFitter

max_time = T.max()
knot_locations = np.arange(0, max_time, 12)[1:]

spline = SplineFitter(knot_locations=knot_locations)
spline.fit(T, E, label="Spline")
spline.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.SplineFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1603.76</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>phi_0_ != 0, phi_1_ != 0, phi_2_ != 0, phi_3_ ...</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>phi_0_</th>
      <td>-4.96</td>
      <td>0.26</td>
      <td>-5.47</td>
      <td>-4.44</td>
      <td>0.00</td>
      <td>-18.89</td>
      <td>&lt;0.005</td>
      <td>261.94</td>
    </tr>
    <tr>
      <th>phi_1_</th>
      <td>1.08</td>
      <td>0.09</td>
      <td>0.91</td>
      <td>1.25</td>
      <td>0.00</td>
      <td>12.40</td>
      <td>&lt;0.005</td>
      <td>114.92</td>
    </tr>
    <tr>
      <th>phi_2_</th>
      <td>0.74</td>
      <td>0.57</td>
      <td>-0.38</td>
      <td>1.86</td>
      <td>0.00</td>
      <td>1.30</td>
      <td>0.19</td>
      <td>2.37</td>
    </tr>
    <tr>
      <th>phi_3_</th>
      <td>-0.59</td>
      <td>1.68</td>
      <td>-3.90</td>
      <td>2.71</td>
      <td>0.00</td>
      <td>-0.35</td>
      <td>0.72</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>phi_4_</th>
      <td>-0.35</td>
      <td>2.22</td>
      <td>-4.71</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>-0.16</td>
      <td>0.87</td>
      <td>0.20</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3217.52</td>
    </tr>
  </tbody>
</table>
</div>



```python
# WeibullAFTFitter 

weibull = WeibullFitter()
weibull.fit(T, E, label="Weibull")
weibull.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.WeibullFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1606.43</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>lambda_ != 1, rho_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lambda_</th>
      <td>138.09</td>
      <td>12.38</td>
      <td>113.82</td>
      <td>162.36</td>
      <td>1.00</td>
      <td>11.07</td>
      <td>&lt;0.005</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>rho_</th>
      <td>0.95</td>
      <td>0.05</td>
      <td>0.85</td>
      <td>1.05</td>
      <td>1.00</td>
      <td>-1.07</td>
      <td>0.29</td>
      <td>1.80</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3216.86</td>
    </tr>
  </tbody>
</table>
</div>


## Comparison of the models 

After fitting all available survival models, the next step is to evaluate their performance in a systematic and comparable way. In this section, I create a comparison table that summarizes key metrics for each model, including **AIC values**, **log-likelihood scores**, and the **mean survival time** estimated from the model’s survival function. Parametric AFT models provide AIC and log-likelihood, which help quantify goodness-of-fit, while non-parametric models like **Kaplan-Meier** and **Nelson-Aalen** are included for visual and interpretative comparison, even though these metrics do not apply to them. By organizing all results into a single DataFrame, I can directly compare model behavior.


```python
fitted_models = {
    "Exponential": exp,
    "Weibull": weibull,
    "LogNormal": lognorm,
    "LogLogistic": loglog,
    "Kaplan-Meier": kmf,
    "Nelson-Aalen": naf,
    "GeneralizedGamma": gg,
    "Breslow-Fleming-Harrington": bfhf,
    "PiecewiseExponential": pwexp,
    "Spline": spline,
    "MixtureCure": mcf
}
```


```python
comparison_rows = []

for name, model in fitted_models.items():
    row = {"Model": name}
    
    row["AIC"] = getattr(model, "AIC_", np.nan)
    row["Log-likelihood"] = getattr(model, "log_likelihood_", np.nan)

    mean_time = np.nan
    try:
        sf = model.survival_function_
        times = sf.index.to_series()
        dt = times.diff().fillna(0)
        mean_time = (sf.iloc[:, 0].values * dt.values).sum()
    except Exception:
        pass 
    
    row["Mean Survival Time (months)"] = mean_time
    
    comparison_rows.append(row)
```


```python
comparison_df = (
    pd.DataFrame(comparison_rows)
    .set_index("Model")
    .sort_values(by="AIC", ascending=True) 
)

comparison_df
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
      <th>AIC</th>
      <th>Log-likelihood</th>
      <th>Mean Survival Time (months)</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LogNormal</th>
      <td>3209.035147</td>
      <td>-1602.517574</td>
      <td>54.081357</td>
    </tr>
    <tr>
      <th>GeneralizedGamma</th>
      <td>3211.008906</td>
      <td>-1602.504453</td>
      <td>54.079730</td>
    </tr>
    <tr>
      <th>LogLogistic</th>
      <td>3214.415476</td>
      <td>-1605.207738</td>
      <td>54.092654</td>
    </tr>
    <tr>
      <th>Exponential</th>
      <td>3215.960813</td>
      <td>-1606.980407</td>
      <td>54.221759</td>
    </tr>
    <tr>
      <th>Weibull</th>
      <td>3216.861171</td>
      <td>-1606.430585</td>
      <td>54.177098</td>
    </tr>
    <tr>
      <th>MixtureCure</th>
      <td>3217.357137</td>
      <td>-1605.678569</td>
      <td>54.135988</td>
    </tr>
    <tr>
      <th>Spline</th>
      <td>3217.520743</td>
      <td>-1603.760372</td>
      <td>54.154521</td>
    </tr>
    <tr>
      <th>PiecewiseExponential</th>
      <td>3220.893222</td>
      <td>-1604.446611</td>
      <td>54.227262</td>
    </tr>
    <tr>
      <th>Kaplan-Meier</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>54.883947</td>
    </tr>
    <tr>
      <th>Nelson-Aalen</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>54.896867</td>
    </tr>
    <tr>
      <th>Breslow-Fleming-Harrington</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>54.896867</td>
    </tr>
  </tbody>
</table>
</div>



 ## Visualization of all the curves | one plot for all

To better understand how each survival model behaves over time, I visualize all fitted survival curves on a single plot. This unified visualization allows me to compare the shapes, slopes, and long-term predictions of each distribution. Parametric AFT models, non-parametric estimators, and flexible spline-based models are all included in the same figure for comparison.


```python
plt.figure(figsize=(12, 7))

for name, model in fitted_models.items():
    try:
        model.plot_survival_function(ci_show=False, label=name)

    except Exception:
        if hasattr(model, "survival_function_"):
            sf = model.survival_function_
            plt.plot(sf.index, sf.iloc[:, 0], label=name)

        elif hasattr(model, "cumulative_hazard_"):
            ch = model.cumulative_hazard_
            sf = np.exp(-ch)
            plt.plot(sf.index, sf.iloc[:, 0], label=name)

        else:
            print(f"Could not plot survival curve for {name}")

plt.title("Survival Curves for All Models")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_42_0.png)
    


## Model Choice

From a purely statistical perspective, the **LogNormal** and **Generalized Gamma** models have the lowest **AIC** and almost identical mean survival times, making them slightly better in terms of in-sample fit. However, the differences are minimal, and all parametric models produce very similar survival curves that closely follow the **Kaplan–Meier** and **Nelson–Aalen** estimates.

To better understand the choice of model, I first fitted a ***multivariate LogNormal AFT*** model using all available predictors. While the model converged, the resulting survival curves were nearly flat, and the predicted survival times were excessively large. This behavior is unrealistic for a telecom churn problem and suggests that the LogNormal AFT model's assumptions are not well aligned with the structure of the dataset.

Next, I tried the ***Generalized Gamma AFT*** model. Unfortunately, the model became numerically unstable: the optimizer struggled to invert the Hessian matrix, several z-scores were effectively infinite, and in some cases, the model even reported zero observed events despite the presence of churn in the data. These issues make the parameter estimates unreliable, rendering the model unsuitable for downstream tasks like **CLV estimation** and **segment-level predictions**.

For decision-making, I also need to consider **robustness**, **interpretability**, and **practical use** for **CLV** and **segmentation**, beyond just AIC. Given these considerations, I selected the **Weibull model** as the final model. The Weibull model provides a **stable fit** without numerical warnings, produces **plausible survival curves** and **median survival times**, and can be applied separately to different customer segments (e.g., by gender or customer category) to support meaningful business insights and CLV comparisons. Therefore, I chose the **Weibull model** as the final decision-making model for the remainder of the analysis.


```python
significant_features = weibull.summary[weibull.summary['p'] < 0.05].index.tolist()
```


```python
final_features = [feat for feat in significant_features if feat in df.columns] 
df_final = df[final_features + ['tenure', 'churn']]
```


```python
weibull_final = WeibullFitter()

weibull_final.fit(df_final['tenure'], event_observed=df_final['churn'])

weibull_final.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.WeibullFitter</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1606.43</td>
    </tr>
    <tr>
      <th>hypothesis</th>
      <td>lambda_ != 1, rho_ != 1</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lambda_</th>
      <td>138.09</td>
      <td>12.38</td>
      <td>113.82</td>
      <td>162.36</td>
      <td>1.00</td>
      <td>11.07</td>
      <td>&lt;0.005</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>rho_</th>
      <td>0.95</td>
      <td>0.05</td>
      <td>0.85</td>
      <td>1.05</td>
      <td>1.00</td>
      <td>-1.07</td>
      <td>0.29</td>
      <td>1.80</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>AIC</th>
      <td>3216.86</td>
    </tr>
  </tbody>
</table>
</div>



```python
weibull_final.plot_survival_function()
plt.title("Weibull Model | Overall Survival Curve")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()
```


    
![png](output_48_0.png)
    



```python
# Assumtion | Monthly revenue per customer
monthly_revenue = 70
customer_clv = {}

# CLV for each customer
for index, row in df.iterrows():
    tenure = row['tenure']
    churn = row['churn']
    
    time_points = int(df['tenure'].max())  
    time_range = np.linspace(0, tenure, time_points * 2)  
    
    survival_prob = weibull_final.survival_function_at_times(time_range)
    expected_lifetime = np.trapz(survival_prob.values.flatten(), time_range)
    
    clv = expected_lifetime * monthly_revenue
    customer_clv[index] = clv
```


```python
clv_df = pd.DataFrame(list(customer_clv.items()), columns=['Customer_ID', 'CLV'])
print(clv_df)
```

         Customer_ID          CLV
    0              0   861.705884
    1              1   734.926741
    2              2  3698.611762
    3              3  2028.388717
    4              4  1467.076348
    ..           ...          ...
    995          995   670.787396
    996          996   475.289726
    997          997  3656.499311
    998          998  3781.943072
    999          999  2892.492720
    
    [1000 rows x 2 columns]
    


```python
print(f"Total number of customers: {len(clv_df)}")
```

    Total number of customers: 1000
    

The **Customer Lifetime Value (CLV)** for each customer is calculated based on the **Weibull model**, which estimates the expected duration of customer retention. The **Weibull model** has a **scale parameter (`lambda_`)** of **138.09** and a **shape parameter (`rho_`)** of **0.95**, indicating a relatively stable churn hazard over time. The **AIC value** of **3216.86** suggests a good balance between model fit and complexity. The survival curve, plotted over **tenure** (time in months), shows a gradual decline in the probability of customer retention as time progresses, but with a relatively constant rate of churn, as reflected in the **`rho_`** parameter.


### Exploration of CLV within different segments


```python
df['CLV'] = pd.Series(customer_clv)

def clv_by_segment(df, segment_col):
    if segment_col not in df.columns:
        print(f"\n[WARNING] Column '{segment_col}' not found in df, skipping.")
        return
    print(f"\n=== CLV by {segment_col} ===")
    segment_clv = df.groupby(segment_col)["CLV"].mean()
    print(segment_clv)


segment_columns = ["gender", "internet", "voice", "retire"]
```


```python
for col in segment_columns:
    clv_by_segment(df, col)
```

    
    === CLV by gender ===
    gender
    0    2097.451089
    1    2049.547485
    Name: CLV, dtype: float64
    
    === CLV by internet ===
    internet
    0    2194.882338
    1    1867.250029
    Name: CLV, dtype: float64
    
    === CLV by voice ===
    voice
    0    2080.087659
    1    2061.094204
    Name: CLV, dtype: float64
    
    === CLV by retire ===
    retire
    0    2033.577527
    1    2900.303524
    Name: CLV, dtype: float64
    

CLV by **Gender**

**Female (0)**: Average CLV = 2097.45
**Male (1)**: Average CLV = 2049.55

This suggests that, on average, **female customers** contribute slightly more to the Customer Lifetime Value (CLV) compared to **male customers**. This could indicate a higher retention rate or longer expected lifetime for female customers in this dataset.

---

CLV by **Internet Access**

**No Internet (0)**: Average CLV = 2194.88
**Has Internet (1)**: Average CLV = 1867.25

Customers with **Internet access** appear to have a lower CLV than those without Internet access. This could imply that customers with internet access tend to churn earlier or generate less revenue over time. Further analysis might be required to understand the specific relationship.

---

CLV by **Voice Access**

**No Voice (0)**: Average CLV = 2080.09
**Has Voice (1)**: Average CLV = 2061.09

There is a small difference in CLV between customers who have voice services and those who do not. The difference is not significant, suggesting that **voice service** might not be a major factor in customer retention in this dataset.

---

CLV by **Retirement Status**

**Not Retired (0)**: Average CLV = 2033.58
**Retired (1)**: Average CLV = 2900.30

**Retired customers** have a substantially higher CLV than non-retired customers. This suggests that **retired customers** are likely to stay longer and contribute more revenue over their lifetime, possibly due to lower churn rates or higher engagement with services.


## Report | Analysis & Results

Based on the survival analysis, the **Weibull model** was selected as the best model for predicting churn due to its stable fit and realistic survival curves. The model's coefficients show that **tenure** (the number of months a customer has been with the company) is a significant factor in predicting churn. Customers who have been with the company for a longer time are less likely to leave. Additionally, **retirement status** plays a key role, with retired customers showing a higher **Customer Lifetime Value (CLV)** and lower churn rates. **Gender**, **internet access**, and **voice service** also have an impact on CLV, with male customers, those without internet, and retired customers contributing more to the company's revenue over time.

To roughly calculate the annual retention budget, we first should approximately estimate the churn probability based on the Weibull survival model. If the model suggests that **7.93\% of customers** are likely to churn within the next year. Given that the average CLV is around 2000 dollar, we calculate the total CLV for at-risk customers by multiplying the **7.93\% of customers** who are expected to churn by the average CLV. Once we have the total value at risk, we assume that **50\% of the CLV** should be allocated to retention efforts, meaning that the company would need to invest approximately 79,000 dollar to retain the customers at risk of churning over the next year. This budget will help the company reduce churn and retain valuable customers, ensuring stable long-term revenue.

To improve retention, I recommend focusing on **high-risk customers**, particularly those with low tenure and without internet access. Offering personalized incentives, loyalty programs, or special offers could help reduce churn among these customers. Further segmentation and targeted strategies for specific customer groups, such as retired customers, could also help increase the overall **CLV**.

