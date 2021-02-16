import pandas as pd

#### 读取概率文件
adam = pd.read_csv('../ADAM/ADAM-p.csv')
max2020 = pd.read_csv('../Max2020/dmj-4935.csv')
aicbic = pd.read_csv('../AICBIC/result1208_p.csv')

#### 后处理后 概率和不为1 融合前需要正则化
max2020['sum'] = max2020.iloc[:,1:].sum(axis=1)
aicbic['sum'] = aicbic.iloc[:,1:].sum(axis=1)
for col in ['label=-1', 'label=0', 'label=1']:
    max2020[col] /= max2020['sum']
for col in ['negative_one', 'zero', 'one']:
    aicbic[col] /= aicbic['sum']


#### 版本1 
tmp = 0.1*adam[['-1','0','1']].values + 0.9*max2020[['label=-1', 'label=0', 'label=1']].values
sub = pd.DataFrame()
sub['cust_no'] = adam.cust_no
sub['label'] = tmp.argmax(axis=1)-1
sub.to_csv('sub_v1.csv',index=False)


#### 版本2
tmp = 0.07*adam[['-1','0','1']].values + 0.9*max2020[['label=-1', 'label=0', 'label=1']].values+0.03*aicbic[['negative_one', 'zero', 'one']].values
sub = pd.DataFrame()
sub['cust_no'] = adam.cust_no
sub['label'] = tmp.argmax(axis=1)-1
sub.to_csv('sub_v2.csv',index=False)