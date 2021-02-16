#### 导入包
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit,StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold,train_test_split,GroupShuffleSplit,StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error,log_loss,confusion_matrix,accuracy_score,cohen_kappa_score
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from tqdm.notebook import tqdm
from collections import Counter
import json
import joblib
import multiprocessing
import time
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import optuna
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)


#### 读取数据

#### 设置路径

path = '../raw_data/'

cust_avli_q3 = pd.read_csv(path+'x_train/cust_avli_Q3.csv')
cust_avli_q4 = pd.read_csv(path+'x_train/cust_avli_Q4.csv')
cust_info_q3 = pd.read_csv(path+'x_train/cust_info_q3.csv')
cust_info_q4 = pd.read_csv(path+'x_train/cust_info_q4.csv')
cust_avli_q1 = pd.read_csv(path+'x_test/cust_avli_Q1.csv')
cust_info_q1 = pd.read_csv(path+'x_test/cust_info_q1.csv')
y_train_q3 = pd.read_csv('../raw_data/y_train_3/y_Q3_3.csv')
y_train_q4 = pd.read_csv('../raw_data/y_train_3/y_Q4_3.csv')
big_event_Q3 = pd.read_csv(path+'x_train/big_event_train/big_event_Q3.csv')
big_event_Q4 = pd.read_csv(path+'x_train/big_event_train/big_event_Q4.csv')
big_event_Q1 = pd.read_csv(path+'x_test/big_event_test/big_event_Q1.csv')
aum_m7 = pd.read_csv(path+'x_train/aum_train/aum_m7.csv') 
aum_m8 = pd.read_csv(path+'x_train/aum_train/aum_m8.csv')
aum_m9 = pd.read_csv(path+'x_train/aum_train/aum_m9.csv')
aum_m10 = pd.read_csv(path+'x_train/aum_train/aum_m10.csv')
aum_m11 = pd.read_csv(path+'x_train/aum_train/aum_m11.csv')
aum_m12 = pd.read_csv(path+'x_train/aum_train/aum_m12.csv')
aum_m1 = pd.read_csv(path+'x_test/aum_test/aum_m1.csv')
aum_m2 = pd.read_csv(path+'x_test/aum_test/aum_m2.csv')
aum_m3 =pd.read_csv(path+'x_test/aum_test/aum_m3.csv')
behavior_m7 = pd.read_csv(path+'x_train/behavior_train/behavior_m7.csv') 
behavior_m8 = pd.read_csv(path+'x_train/behavior_train/behavior_m8.csv')
behavior_m9 = pd.read_csv(path+'x_train/behavior_train/behavior_m9.csv')
behavior_m10 = pd.read_csv(path+'x_train/behavior_train/behavior_m10.csv')
behavior_m11 = pd.read_csv(path+'x_train/behavior_train/behavior_m11.csv')
behavior_m12 = pd.read_csv(path+'x_train/behavior_train/behavior_m12.csv')
behavior_m1 = pd.read_csv(path+'x_test/behavior_test/behavior_m1.csv')
behavior_m2 = pd.read_csv(path+'x_test/behavior_test/behavior_m2.csv')
behavior_m3 =pd.read_csv(path+'x_test/behavior_test/behavior_m3.csv')
cunkuan_m7 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m7.csv') 
cunkuan_m8 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m8.csv')
cunkuan_m9 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m9.csv')
cunkuan_m10 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m10.csv')
cunkuan_m11 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m11.csv')
cunkuan_m12 = pd.read_csv(path+'x_train/cunkuan_train/cunkuan_m12.csv')
cunkuan_m1 = pd.read_csv(path+'x_test/cunkuan_test/cunkuan_m1.csv')
cunkuan_m2 = pd.read_csv(path+'x_test/cunkuan_test/cunkuan_m2.csv')
cunkuan_m3 =pd.read_csv(path+'x_test/cunkuan_test/cunkuan_m3.csv')


#### 只取有效客户数据
cust_info_q3 = cust_info_q3.merge(cust_avli_q3,how='inner',on='cust_no')
print(cust_info_q3.shape)
cust_info_q4 = cust_info_q4.merge(cust_avli_q4,how='inner',on='cust_no')
print(cust_info_q4.shape)
cust_info_q1 = cust_info_q1.merge(cust_avli_q1,how='inner',on='cust_no')
print(cust_info_q1.shape)
big_event_Q3 = big_event_Q3.merge(cust_avli_q3,how='inner',on='cust_no')
print(big_event_Q3.shape)
big_event_Q4 = big_event_Q4.merge(cust_avli_q4,how='inner',on='cust_no')
print(big_event_Q4.shape)
big_event_Q1 = big_event_Q1.merge(cust_avli_q1,how='inner',on='cust_no')
print(big_event_Q1.shape)
cust_info_q3 = cust_info_q3.merge(cust_avli_q3,how='inner',on='cust_no')
print(cust_info_q3.shape)
cust_info_q4 = cust_info_q4.merge(cust_avli_q4,how='inner',on='cust_no')
print(cust_info_q4.shape)
cust_info_q1 = cust_info_q1.merge(cust_avli_q1,how='inner',on='cust_no')
print(cust_info_q1.shape)
print('all avli q3',cust_avli_q3.shape[0])
aum_m7 = aum_m7.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m7',aum_m7.shape[0])
aum_m8 = aum_m8.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m8',aum_m8.shape[0])
aum_m9 = aum_m9.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m9',aum_m9.shape[0])
print('all avli q4',cust_avli_q4.shape[0])
aum_m10 = aum_m10.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m10',aum_m10.shape[0])
aum_m11 = aum_m11.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m11',aum_m11.shape[0])
aum_m12 = aum_m12.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m12',aum_m12.shape[0])
print('all avli q1',cust_avli_q1.shape[0])
aum_m1 = aum_m1.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m1',aum_m1.shape[0])
aum_m2 = aum_m2.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m2',aum_m2.shape[0])
aum_m3 = aum_m3.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m3',aum_m3.shape[0])
print('all avli q3',cust_avli_q3.shape[0])
behavior_m7 = behavior_m7.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m7',behavior_m7.shape[0])
behavior_m8 = behavior_m8.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m8',behavior_m8.shape[0])
behavior_m9 = behavior_m9.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m9',behavior_m9.shape[0])
print('all avli q4',cust_avli_q4.shape[0])
behavior_m10 = behavior_m10.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m10',behavior_m10.shape[0])
behavior_m11 = behavior_m11.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m11',behavior_m11.shape[0])
behavior_m12 = behavior_m12.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m12',behavior_m12.shape[0])
print('all avli q1',cust_avli_q1.shape[0])
behavior_m1 = behavior_m1.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m1',behavior_m1.shape[0])
behavior_m2 = behavior_m2.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m2',behavior_m2.shape[0])
behavior_m3 = behavior_m3.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m3',behavior_m3.shape[0])
print('all avli q3',cust_avli_q3.shape[0])
cunkuan_m7 = cunkuan_m7.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m7',cunkuan_m7.shape[0])
cunkuan_m8 = cunkuan_m8.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m8',cunkuan_m8.shape[0])
cunkuan_m9 = cunkuan_m9.merge(cust_avli_q3,how='inner',on = 'cust_no')
print('m9',cunkuan_m9.shape[0])
print('all avli q4',cust_avli_q4.shape[0])
cunkuan_m10 = cunkuan_m10.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m10',cunkuan_m10.shape[0])
cunkuan_m11 = cunkuan_m11.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m11',cunkuan_m11.shape[0])
cunkuan_m12 = cunkuan_m12.merge(cust_avli_q4,how='inner',on = 'cust_no')
print('m12',cunkuan_m12.shape[0])
print('all avli q1',cust_avli_q1.shape[0])
cunkuan_m1 = cunkuan_m1.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m1',cunkuan_m1.shape[0])
cunkuan_m2 = cunkuan_m2.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m2',cunkuan_m2.shape[0])
cunkuan_m3 = cunkuan_m3.merge(cust_avli_q1,how='inner',on = 'cust_no')
print('m3',cunkuan_m3.shape[0])
train_q3 = y_train_q3.copy()
train_q3['quarter'] = 3
train_q4 = y_train_q4.copy()
train_q4['quarter'] = 4
test_q1 = cust_avli_q1.copy()
test_q1['quarter'] = 5


### 特征处理
### 处理B6分布不一致
behavior_m3['B6'] = pd.to_datetime(behavior_m3['B6'])
behavior_m9['B6'] = pd.to_datetime(behavior_m9['B6'])
behavior_m12['B6'] = pd.to_datetime(behavior_m12['B6'])
behavior_m9.loc[behavior_m9.B6<pd.to_datetime('2019-07-01 00:00:00'),'B6'] = np.nan
behavior_m12.loc[behavior_m12.B6<pd.to_datetime('2019-10-01 00:00:00'),'B6'] = np.nan
behavior_m3['diff_last_trans'] = (pd.to_datetime('2020-04-01 00:00:00') - pd.to_datetime(behavior_m3['B6'])).dt.days
behavior_m9['diff_last_trans'] = (pd.to_datetime('2019-10-01 00:00:00') - pd.to_datetime(behavior_m9['B6'])).dt.days
behavior_m12['diff_last_trans'] = (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(behavior_m12['B6'])).dt.days
for table in [behavior_m3,behavior_m9,behavior_m12]:
    table.drop(['B6'],axis=1,inplace=True)
### 关联特征
train_q3 = train_q3.merge(aum_m7.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(aum_m8.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(aum_m9.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q4 = train_q4.merge(aum_m10.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(aum_m11.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(aum_m12.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

test_q1 = test_q1.merge(aum_m1.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(aum_m2.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(aum_m3.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q3 = train_q3.merge(behavior_m7.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(behavior_m8.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(behavior_m9.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q4 = train_q4.merge(behavior_m10.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(behavior_m11.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(behavior_m12.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')


test_q1 = test_q1.merge(behavior_m1.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(behavior_m2.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(behavior_m3.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q3 = train_q3.merge(cunkuan_m7.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(cunkuan_m8.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q3 = train_q3.merge(cunkuan_m9.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q4 = train_q4.merge(cunkuan_m10.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(cunkuan_m11.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
train_q4 = train_q4.merge(cunkuan_m12.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')


test_q1 = test_q1.merge(cunkuan_m1.set_index('cust_no').add_prefix('num1_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(cunkuan_m2.set_index('cust_no').add_prefix('num2_').reset_index(),on='cust_no',how='left')
test_q1 = test_q1.merge(cunkuan_m3.set_index('cust_no').add_prefix('num3_').reset_index(),on='cust_no',how='left')

train_q3 = train_q3.merge(cust_info_q3,on='cust_no',how='left')
train_q4 = train_q4.merge(cust_info_q4,on='cust_no',how='left')
test_q1 = test_q1.merge(cust_info_q1,on='cust_no',how='left')

tmp = pd.DataFrame()
tmp['cust_no'] = train_q3.cust_no
tmp['last_label'] = train_q3.label
train_q4 = train_q4.merge(tmp,on='cust_no',how='left')
train_q4.last_label = train_q4.last_label.fillna(-999)

tmp = pd.DataFrame()
tmp['cust_no'] = train_q4.cust_no
tmp['last_label'] = train_q4.label
test_q1 = test_q1.merge(tmp,on='cust_no',how='left')
test_q1.last_label = test_q1.last_label.fillna(-999)
train = train_q3.append(train_q4,ignore_index=True)

#### 每个月存入总金额的和/三个月平均
cunru_list = [f'X{_}' for _ in range(1,9) if _ !=7]
for prefix in ['num1','num2','num3']:
    col_lists = [prefix+'_'+_ for _ in cunru_list]
    train[f'{prefix}_cunru_sum'] = train[col_lists].sum(axis=1)
    test_q1[f'{prefix}_cunru_sum'] = test_q1[col_lists].sum(axis=1)
train['cunru_month_avg'] = train[['num1_cunru_sum','num2_cunru_sum','num3_cunru_sum']].mean(axis=1)
test_q1['cunru_month_avg'] = test_q1[['num1_cunru_sum','num2_cunru_sum','num3_cunru_sum']].mean(axis=1)


#### 3和2的差 3和1的差 2和1的差
for i in [1,2,3]:
    for j in [1,2,3]:
        if i>j:
            train[f'diff_cunru_sum_{i}_{j}'] = train[f'num{i}_cunru_sum'] - train[f'num{j}_cunru_sum']
            train[f'ratio_cunru_sum_{i}_{j}'] = train[f'num{i}_cunru_sum'] / train[f'num{j}_cunru_sum']
            test_q1[f'diff_cunru_sum_{i}_{j}'] = test_q1[f'num{i}_cunru_sum'] - test_q1[f'num{j}_cunru_sum']
            test_q1[f'ratio_cunru_sum_{i}_{j}'] = test_q1[f'num{i}_cunru_sum'] / test_q1[f'num{j}_cunru_sum']
#####最后一个月和前3个月的均值
train['ratio_num3_cunru_sum_mean'] = train['num3_cunru_sum']/train['cunru_month_avg']
test_q1['ratio_num3_cunru_sum_mean'] = test_q1['num3_cunru_sum']/test_q1['cunru_month_avg']

#####最后一个月和前3个月的差值
train['diff_num3_cunru_sum_mean'] = train['num3_cunru_sum'] - train['cunru_month_avg']
test_q1['diff_num3_cunru_sum_mean'] = test_q1['num3_cunru_sum'] - test_q1['cunru_month_avg']

#### 每个月贷款占总存款的比例
for prefix in ['num1','num2','num3']:
    train[prefix+'_ratio_daikuan_cunkuan'] = train[prefix+'_X7']/train[prefix+'_cunru_sum']
    test_q1[prefix+'_ratio_daikuan_cunkuan'] = test_q1[prefix+'_X7']/test_q1[prefix+'_cunru_sum']
    train[prefix+'_diff_daikuan_cunkuan'] = train[prefix+'_cunru_sum'] - train[prefix+'_X7'] 
    test_q1[prefix+'_diff_daikuan_cunkuan'] = test_q1[prefix+'_cunru_sum'] - test_q1[prefix+'_X7']
#### 贷款比例的平均
train['avg_daikuan_sum_ratio'] = train[['num1_ratio_daikuan_cunkuan','num2_ratio_daikuan_cunkuan','num3_ratio_daikuan_cunkuan']].mean(axis=1)
test_q1['avg_daikuan_sum_ratio'] = test_q1[['num1_ratio_daikuan_cunkuan','num2_ratio_daikuan_cunkuan','num3_ratio_daikuan_cunkuan']].mean(axis=1)
### 存款 - 贷款的平均
train['avg_daikuan_sum_diff'] = train[['num1_diff_daikuan_cunkuan','num2_diff_daikuan_cunkuan','num3_diff_daikuan_cunkuan']].mean(axis=1)
test_q1['avg_daikuan_sum_diff'] = test_q1[['num1_diff_daikuan_cunkuan','num2_diff_daikuan_cunkuan','num3_diff_daikuan_cunkuan']].mean(axis=1)

#### 3个月贷款金额的平均
train['daikuan_month_avg'] = train[['num1_X7','num2_X7','num3_X7']].mean(axis=1)
test_q1['daikuan_month_avg'] = test_q1[['num1_X7','num2_X7','num3_X7']].mean(axis=1)
#### 3和2的差 3和1的差 2和1的差 贷款
for i in [1,2,3]:
    for j in [1,2,3]:
        if i>j:
            train[f'diff_daikuan_{i}_{j}'] = train[f'num{i}_X7'] - train[f'num{j}_X7']
            train[f'ratio_daikuan_{i}_{j}'] = train[f'num{i}_X7'] / train[f'num{j}_X7']
            test_q1[f'diff_daikuan_{i}_{j}'] = test_q1[f'num{i}_X7'] - test_q1[f'num{j}_X7']
            test_q1[f'ratio_daikuan_{i}_{j}'] = test_q1[f'num{i}_X7'] / test_q1[f'num{j}_X7']
#####最后一个月和前3个月的均值/差
train['ratio_num3_daikuan_mean'] = train['num3_X7']/train['daikuan_month_avg']
test_q1['ratio_num3_daikuan_mean'] = test_q1['num3_X7']/test_q1['daikuan_month_avg']

train['diff_num3_daikuan_mean'] = train['num3_X7']- train['daikuan_month_avg']
test_q1['diff_num3_daikuan_mean'] = test_q1['num3_X7']- test_q1['daikuan_month_avg']

#### 差分
all_df = train.append(test_q1,ignore_index=True)
all_df['diff_quarter_num3_cunru_sum'] = all_df.groupby(['cust_no'])['num3_cunru_sum'].diff()
all_df.sort_index(inplace=True)

train_new = all_df.iloc[:145296,:].copy()
test_new = all_df.iloc[145296:,:].copy()
test_new.drop(['label'],axis=1,inplace=True)

#### train = train_q3.append(train_q4,ignore_index=True)
y_train = train_new.label
X_train = train_new.drop(['cust_no','quarter','label'],axis=1)

X_test = test_new.drop(['cust_no','quarter'],axis=1)
y_train = y_train +1

### Label encoding
cat_list = ['I1','I3','I5','I8','I10','I12','I13','I14']
for feature in cat_list:
    label_encod = LabelEncoder()
    label_encod.fit(list(X_train[feature].astype(str).values) + list(X_test[feature].astype(str).values))
    X_train[feature] = label_encod.transform(list(X_train[feature].astype(str).values))
    X_test[feature] = label_encod.transform(list(X_test[feature].astype(str).values))

print(X_train.shape)
print(X_test.shape)

### I3 target encoding
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)

# for col in ['I3']:#,'I5','I10','I13','I14']:
#     for val in X_train.I3.unique():
#         X_train[col+'_'+str(val)+'_'+'target_enc'] = -999
for train_index,test_index in skf.split(X_train,y_train):
    X_train2 = X_train.iloc[train_index,:]
    y_train2 = y_train.iloc[train_index]
    for col in ['I3']:
        
        tmp = X_train2.copy()
        tmp['label'] = y_train2
        tmp1 = tmp.groupby([col,'label'])['I4'].size().reset_index()
        tmp1 = pd.pivot(tmp1,index='I3',columns='label')
        new_col = []
        for a,b in tmp1.columns:
            new_col.append(b)
        tmp1.columns = new_col
        tmp1.index.name=None
        tmp1['sum'] = tmp1.sum(axis=1)
        for _ in tmp1.columns:
            tmp1[_] = tmp1[_]/tmp1['sum']
        tmp1.drop(['sum'],axis=1,inplace=True)
        for val1 in tmp1.columns:
            X_train.loc[test_index,col+'_'+str(val1)+'_'+'target_enc'] = X_train.loc[test_index,col].map(tmp1[val1])

for col in ['I3']:
        tmp = X_train.copy()
        tmp['label'] = y_train
        tmp1 = tmp.groupby([col,'label'])['I4'].size().reset_index()
        tmp1 = pd.pivot(tmp1,index='I3',columns='label')
        new_col = []
        for a,b in tmp1.columns:
            new_col.append(b)
        tmp1.columns = new_col
        tmp1.index.name=None
        tmp1['sum'] = tmp1.sum(axis=1)
        for _ in tmp1.columns:
            tmp1[_] = tmp1[_]/tmp1['sum']
        tmp1.drop(['sum'],axis=1,inplace=True)
        for val1 in tmp1.columns:
            X_test[col+'_'+str(val1)+'_'+'target_enc'] = X_test[col].map(tmp1[val1])

### 填补空值

X_train.fillna(-999,inplace=True)
X_test.fillna(-999,inplace=True)

print(X_train.shape)
print(X_test.shape)

### Lightgbm参数
params = {'bagging_freq':1,
'num_leaves': 370,
 'subsample': 0.9967584074071576,
 'colsample_bytree': 0.9134179494490106,
 'min_child_weight': 0.5400296245490022,
 'min_child_samples': 30,
 'reg_alpha': 2.7153164704173993,
 'reg_lambda': 1.2598378046329661,
 'min_split_gain': 0.15398430096611754}

from numba import njit
@njit
def qwk3(a1, a2, max_rat=2):

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  1 if i!=j else 0

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (1 if i!=j else 0)

    e = e / a1.shape[0]

    return 1 - o / e

def kappa_custom(y_true,y_pred):
    y_pred = np.reshape(y_pred,(-1,3),'F')
    y_pred = y_pred.argmax(axis=1)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    return 'kappa', qwk3(y_true,y_pred), True

### 模型训练

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
fea_impor = 0
oof_train = np.zeros((X_train.shape[0],3))
y_pred = np.zeros((X_test.shape[0],3))
kappa_fold_list = []
k = 0
for train_index,test_index in skf.split(X_train,y_train):
    k+=1
    print(f'{k}folds begins******************************')
    X_train2 = X_train.iloc[train_index,:]
    y_train2 = y_train.iloc[train_index]
    X_test2 = X_train.iloc[test_index,:]
    y_test2 = y_train.iloc[test_index]
    clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47,learning_rate=0.01,importance_type = 'gain',metric = 'None',
                 n_jobs = -1,**params,class_weight={0:2.063381086753603,1:1.9261633984493038,2:1})
    # clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47+_,learning_rate=0.01,importance_type = 'gain',
    #                     n_jobs = -1,num_leaves=20,bagging_freq=1,colsample_bytree=0.5,subsample=1,min_child_weight = 0.1,
    #                             min_child_samples = 250,reg_alpha = 1.5,reg_lambda = 1,metric = 'None',min_split_gain = 0)
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=200,verbose=50,\
            eval_metric=kappa_custom)
    tmp = clf.predict_proba(X_test2)
    oof_train[test_index,:] = tmp
    kappa_loss = cohen_kappa_score(y_test2,tmp.argmax(axis=1))
    print(f'fold{k} kappa_loss',kappa_loss)
    kappa_fold_list.append(kappa_loss)
    y_pred += clf.predict_proba(X_test)/skf.n_splits
    fea_impor += clf.feature_importances_/skf.n_splits
for _ in kappa_fold_list:
    print(_)
print('mean kappa',np.mean(kappa_fold_list))
print('oof kappa',cohen_kappa_score(y_train,oof_train.argmax(axis=1)))

#### 生成每个样本的概率文件 方便融合
sub = pd.DataFrame()
sub['cust_no'] = test_q1.cust_no
for num,col in enumerate(['-1','0','1']):
    sub[col] = y_pred[:,num]
sub.to_csv('ADAM-p.csv',index=False)

#### 生成每个样本的提交文件
sub = pd.DataFrame()
sub['cust_no'] = test_q1.cust_no
sub['label'] = y_pred.argmax(axis=1) - 1
sub.to_csv('ADAM.csv',index=False)