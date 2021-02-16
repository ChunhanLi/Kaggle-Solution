# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:44:45 2020

@author: 31058
"""
import os
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
# 数据路径
os.chdir(r'D:/competition/2020_10_30/给选手数据')
class deal_data():
    """
    处理原始数据，转化为可以入模的数据类型。
    """
    def __init__(self):
        self.y_label_q3 = pd.read_csv(r'./y_train_3/y_Q3_3.csv')
        self.y_label_q4 = pd.read_csv(r'./y_train_3/y_Q4_3.csv')
        self.test_q1 = pd.read_csv(r'./x_test/cust_avli_Q1.csv')
        
    def deal_aum(self):
        """
        作用:处理月末时点资产数据
        """
        train_q3 = self.y_label_q3.copy()
        train_q4 = self.y_label_q4.copy()
        test_q1 = self.test_q1.copy()
        # 处理三季度训练数据
        for month in [7,8,9]:
            aum_tmp = pd.read_csv(r'./x_train/aum_train/aum_m%s.csv'%month)
            aum_tmp = aum_tmp.rename(columns=dict(zip([name for name in aum_tmp.columns if 'X' in name],[name+'_'+str(month-6) for name in aum_tmp.columns if 'X' in name])))
            train_q3 = pd.merge(train_q3,aum_tmp,how='left',on='cust_no')
        # 处理四季度训练数据
        for month in [10,11,12]:
            aum_tmp = pd.read_csv(r'./x_train/aum_train/aum_m%s.csv'%month)
            aum_tmp = aum_tmp.rename(columns=dict(zip([name for name in aum_tmp.columns if 'X' in name],[name+'_'+str(month-9) for name in aum_tmp.columns if 'X' in name])))
            train_q4 = pd.merge(train_q4,aum_tmp,how='left',on='cust_no')
        # 处理一季度测试数据
        for month in [1,2,3]:
            aum_tmp = pd.read_csv(r'./x_test/aum_test/aum_m%s.csv'%month)
            aum_tmp = aum_tmp.rename(columns=dict(zip([name for name in aum_tmp.columns if 'X' in name],[name+'_'+str(month) for name in aum_tmp.columns if 'X' in name])))
            test_q1 = pd.merge(test_q1,aum_tmp,how='left',on='cust_no')
            
        return train_q3, train_q4, test_q1
    
    def deal_behavior(self):
        """
        作用:处理月行为数据
        """
        def behavior_feature(Data):
            """
            作用:构造特征
            """
            Data['B23'] = Data['B3']/Data['B2']
            Data['B45'] = Data['B5']/Data['B4']
            Data['B21'] = Data['B2']/Data['B1']
            Data['B41'] = Data['B4']/Data['B1']
            Data['B53'] = Data['B5']/Data['B3']
            
            return Data
        
        def change_rate(Sers):
            return Sers.tail(1).values
        
        train_q3 = self.y_label_q3.copy()
        train_q4 = self.y_label_q4.copy()
        test_q1 = self.test_q1.copy()
        print("开始处理三季度行为训练数据")
        data_all_q3 = pd.concat([pd.read_csv(r'./x_train/behavior_train/behavior_m%s.csv'%month) for month in [7,8,9]]) # 将各月数据合并
        data_all_q3 = data_all_q3.sort_values(by='cust_no').reset_index(drop=True)
        # 处理时间特征，用最大的时间做差
        data_all_q3['B6'] = (pd.to_datetime(data_all_q3.B6).max()-pd.to_datetime(data_all_q3.B6)).apply(lambda x:x.days if x.days<90 else 90)
        data_all_q3 = behavior_feature(data_all_q3)
        col_list = [name for name in data_all_q3.columns if 'B' in name]
        agg_fun = [change_rate, 'min', 'max', 'mean', 'median', 'std', 'skew']
        agg_stat = dict(zip(col_list,[agg_fun]*len(col_list)))
        # 做组统计特征
        group_aum_q3 = data_all_q3.groupby('cust_no').agg(agg_stat)
        group_aum_q3.columns = [f[0]+'_'+f[1] for f in group_aum_q3.columns]
        group_aum_q3 = group_aum_q3.reset_index()
        train_q3 = train_q3.merge(group_aum_q3, how='left', on='cust_no')
        print("处理四季度行为训练数据")
        data_all_q4 = pd.concat([pd.read_csv(r'./x_train/behavior_train/behavior_m%s.csv'%month) for month in [10,11,12]])
        data_all_q4 = data_all_q4.sort_values(by='cust_no').reset_index(drop=True)
        data_all_q4['B6'] = (pd.to_datetime(data_all_q4.B6).max()-pd.to_datetime(data_all_q4.B6)).apply(lambda x:x.days if x.days<90 else 90)
        data_all_q4 = behavior_feature(data_all_q4)
        group_aum_q4 = data_all_q4.groupby('cust_no').agg(agg_stat)
        group_aum_q4.columns = [f[0]+'_'+f[1] for f in group_aum_q4.columns]
        group_aum_q4 = group_aum_q4.reset_index()
        train_q4 = train_q4.merge(group_aum_q4, how='left', on='cust_no')
        print("开始处理一季度行为测试数据")
        test_all_q1 = pd.concat([pd.read_csv(r'./x_test/behavior_test/behavior_m%s.csv'%month) for month in [1,2,3]])
        test_all_q1 = test_all_q1.sort_values(by='cust_no').reset_index(drop=True)
        test_all_q1['B6'] = (pd.to_datetime(test_all_q1.B6).max()-pd.to_datetime(test_all_q1.B6)).apply(lambda x:x.days if x.days<90 else 90)
        test_all_q1 = behavior_feature(test_all_q1)
        group_aum_q1 = test_all_q1.groupby('cust_no').agg(agg_stat)
        group_aum_q1.columns = [f[0]+'_'+f[1] for f in group_aum_q1.columns]
        group_aum_q1 = group_aum_q1.reset_index()
        test_q1 = test_q1.merge(group_aum_q1, how='left', on='cust_no')
        
        return train_q3, train_q4, test_q1
        
        
    def deal_big_event(self):
        """
        作用:处理季度的客户重大历史数据
        """
        train_q3 = self.y_label_q3.copy()
        train_q4 = self.y_label_q4.copy()
        test_q1 = self.test_q1.copy()
        
        def time_deal(Data):
            """
            作用:处理日期特征
            """
            be_tmp3 = Data.copy()
            # 统计截止时间距离开户日期天数
            be_tmp3['E1_days'] = (pd.to_datetime(be_tmp3.E1).max()-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 开户时间与网银开通时间间隔
            be_tmp3['E2_E1days'] = (pd.to_datetime(be_tmp3.E2)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 开户时间与手机app开通时间间隔
            be_tmp3['E3_E1days'] = (pd.to_datetime(be_tmp3.E3)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一网银登陆时间与网银开通时间间隔
            be_tmp3['E4_E2days'] = (pd.to_datetime(be_tmp3.E3)-pd.to_datetime(be_tmp3.E2)).apply(lambda x:x.days)
            # 第一次手机app登录时间与手机app开通时间间隔
            be_tmp3['E5_E3days'] = (pd.to_datetime(be_tmp3.E5)-pd.to_datetime(be_tmp3.E3)).apply(lambda x:x.days)
            # 第一次活期存款业务时间与开户时间间隔
            be_tmp3['E6_E1days'] = (pd.to_datetime(be_tmp3.E6)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次定期存款时间与开户时间间隔
            be_tmp3['E7_E1days'] = (pd.to_datetime(be_tmp3.E7)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次贷款业务时间与开户日期间隔
            be_tmp3['E8_E1days'] = (pd.to_datetime(be_tmp3.E8)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次逾期时间与第一次贷款时间间隔
            be_tmp3['E9_E8days'] = (pd.to_datetime(be_tmp3.E9)-pd.to_datetime(be_tmp3.E8)).apply(lambda x:x.days)
            # 第一次资金交易与开户间隔
            be_tmp3['E10_E1days'] = (pd.to_datetime(be_tmp3.E10)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次银证转账与开户间隔
            be_tmp3['E11_E1days'] = (pd.to_datetime(be_tmp3.E11)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次柜台转账与开户间隔
            be_tmp3['E12_E1days'] = (pd.to_datetime(be_tmp3.E12)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 第一次网银转账与网银开通
            be_tmp3['E13_E2days'] = (pd.to_datetime(be_tmp3.E13)-pd.to_datetime(be_tmp3.E2)).apply(lambda x:x.days)
            # 第一次手机app转账与手机app开通时间
            be_tmp3['E14_E3days'] = (pd.to_datetime(be_tmp3.E14)-pd.to_datetime(be_tmp3.E3)).apply(lambda x:x.days)
            # 第一次手机app转账与手机app开通时间
            be_tmp3['E14_E5days'] = (pd.to_datetime(be_tmp3.E14)-pd.to_datetime(be_tmp3.E5)).apply(lambda x:x.days)
            # 第一次最大转出他行与开户日期天数
            be_tmp3['E16_E1days'] = (pd.to_datetime(be_tmp3.E16)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            # 最大转入与开户日期天数
            be_tmp3['E18_E1days'] = (pd.to_datetime(be_tmp3.E18)-pd.to_datetime(be_tmp3.E1)).apply(lambda x:x.days)
            
            be_tmp3 = be_tmp3.drop(columns=['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E16','E18'])
            
            return be_tmp3
        be_tmp3 = pd.read_csv(r'./x_train/big_event_train/big_event_Q3.csv')
        train_q3 = pd.merge(train_q3, be_tmp3, how='left', on='cust_no')
        be_tmp4 = pd.read_csv(r'./x_train/big_event_train/big_event_Q4.csv')
        train_q4 = pd.merge(train_q4, be_tmp4, how='left', on='cust_no')
        test_q1 = pd.read_csv(r'./x_test/big_event_test/big_event_Q1.csv')
        train_q3 = time_deal(train_q3)
        train_q4 = time_deal(train_q4)
        test_q1 = time_deal(test_q1)
        
        return train_q3, train_q4, test_q1
    
    def deal_cunkuan(self):
        """
        作用:处理月的存款数据
        """
        train_q3 = self.y_label_q3.copy()
        train_q4 = self.y_label_q4.copy()
        test_q1 = self.test_q1.copy()      
        print("处理第三季度存款训练数据")
        data_all_q3 = pd.concat([pd.read_csv(r'./x_train/cunkuan_train/cunkuan_m%s.csv'%month) for month in [7,8,9]])
        data_all_q3 = data_all_q3.sort_values(by='cust_no').reset_index(drop=True)
        data_all_q3['C3'] = data_all_q3['C1']/data_all_q3['C2']
        col_list = [name for name in data_all_q3.columns if 'C' in name]
        agg_fun = [ 'min', 'max', 'mean', 'median', 'std', 'skew']
        agg_stat = dict(zip(col_list,[agg_fun]*len(col_list)))
        group_aum_q3 = data_all_q3.groupby('cust_no').agg(agg_stat)
        group_aum_q3.columns = [f[0]+'_'+f[1] for f in group_aum_q3.columns]
        group_aum_q3 = group_aum_q3.reset_index()
        train_q3 = train_q3.merge(group_aum_q3, how='left', on='cust_no')
        print("处理第四季度存款数据")
        data_all_q4 = pd.concat([pd.read_csv(r'./x_train/cunkuan_train/cunkuan_m%s.csv'%month) for month in [10,11,12]])
        data_all_q4 = data_all_q4.sort_values(by='cust_no').reset_index(drop=True)
        data_all_q4['C3'] = data_all_q4['C1']/data_all_q4['C2']
        group_aum_q4 = data_all_q4.groupby('cust_no').agg(agg_stat)
        group_aum_q4.columns = [f[0]+'_'+f[1] for f in group_aum_q4.columns]
        group_aum_q4 = group_aum_q4.reset_index()
        train_q4 = train_q4.merge(group_aum_q4, how='left', on='cust_no')
        print("处理第一季度存款测试数据")
        test_all_q1 = pd.concat([pd.read_csv(r'./x_test/cunkuan_test/cunkuan_m%s.csv'%month) for month in [1,2,3]])
        test_all_q1 = test_all_q1.sort_values(by='cust_no').reset_index(drop=True)
        test_all_q1['C3'] = test_all_q1['C1']/test_all_q1['C2']
        group_aum_q1 = test_all_q1.groupby('cust_no').agg(agg_stat)
        group_aum_q1.columns = [f[0]+'_'+f[1] for f in group_aum_q1.columns]
        group_aum_q1 = group_aum_q1.reset_index()
        test_q1 = test_q1.merge(group_aum_q1, how='left', on='cust_no')
        
        return train_q3, train_q4, test_q1
    
    def deal_cust(self):
        """
        作用:处理季度的客户信息
        """
        train_q3 = self.y_label_q3.copy()
        train_q4 = self.y_label_q4.copy()
        #test_q1 = self.test_q1.copy() 
        cust_tmp_q3 = pd.read_csv(r'./x_train/cust_info_q3.csv')
        cust_tmp_q4 = pd.read_csv(r'./x_train/cust_info_q4.csv')
        cust_test = pd.read_csv(r'./x_test/cust_info_q1.csv')
        train_q3 = train_q3.merge(cust_tmp_q3, on='cust_no', how='left')
        train_q4 = train_q4.merge(cust_tmp_q4, on='cust_no', how='left')
        train_all = pd.concat([train_q3, train_q4]) 
        train_all = train_all.sort_values(by=['label'])
        def mean_encoding(col):
            """
            对特征进行mean encoding
            """
            sort_I5 = [f[0] for f in sorted(train_all.groupby(col)['label'].mean().items(), key= lambda item:item[1])]
            I5_dict = dict(zip(sort_I5, range(len(sort_I5))))
            cust_tmp_q3[col] = cust_tmp_q3[col].map(I5_dict)
            cust_tmp_q4[col] = cust_tmp_q4[col].map(I5_dict)
            cust_test[col] = cust_test[col].map(I5_dict)
            
            return cust_tmp_q3, cust_tmp_q4, cust_test
            
        # I5进行mean encoding
        cust_tmp_q3, cust_tmp_q4, cust_test = mean_encoding('I5')
        # I10进行mean encoding
        cust_tmp_q3, cust_tmp_q4, cust_test = mean_encoding('I10')
        # I13进行mean encoding
        cust_tmp_q3, cust_tmp_q4, cust_test = mean_encoding('I13')
        # I14进行mean encoding
        cust_tmp_q3, cust_tmp_q4, cust_test = mean_encoding('I14')
        # 将其余字符型特征encoding
        for col in cust_tmp_q3.select_dtypes('object').columns:
            if col=='cust_no':continue
            cust_tmp_q3[col] = cust_tmp_q3[col].fillna('-1')
            cust_tmp_q4[col] = cust_tmp_q4[col].fillna('-1')
            cust_test[col] = cust_test[col].fillna('-1')
            le = preprocessing.LabelEncoder()
            le.fit(pd.concat([cust_tmp_q3[[col]], cust_tmp_q4[[col]], cust_test[[col]]], axis=0, ignore_index=True))
            cust_tmp_q3[col] = le.transform(cust_tmp_q3[col])
            cust_tmp_q4[col] = le.transform(cust_tmp_q4[col])
            cust_test[col] = le.transform(cust_test[col])
        
        return cust_tmp_q3, cust_tmp_q4, cust_test
    
    def save_data(self):
        print("处理月末时点资产数据")
        train_q3, train_q4, test_q1 = self.deal_aum()
        train_q3.to_csv(r'./x_train/aum_train/train_aum_q3_concat.csv', index=False)
        train_q4.to_csv(r'./x_train/aum_train/train_aum_q4_concat.csv', index=False)
        test_q1.to_csv(r'./x_test/aum_test/test_aum_q1_concat.csv', index=False)
        print("处理月行为数据")
        train_q3, train_q4, test_q1 = self.deal_behavior()
        train_q3.to_csv(r'./x_train/behavior_train/train_behavior_q3.csv', index=False)
        train_q4.to_csv(r'./x_train/behavior_train/train_behavior_q4.csv', index=False)
        test_q1.to_csv(r'./x_test/behavior_test/test_behavior_q1.csv', index=False)
        print("处理季度的客户重大历史数据")
        train_q3, train_q4, test_q1 = self.deal_big_event()
        train_q3.to_csv(r'./x_train/big_event_train/train_be_q3.csv', index=False)
        train_q4.to_csv(r'./x_train/big_event_train/train_be_q4.csv', index=False)
        test_q1.to_csv(r'./x_test/big_event_test/test_be_q4.csv', index=False)
        print("处理月的存款数据")
        train_q3, train_q4, test_q1 = self.deal_cunkuan()
        train_q3.to_csv(r'./x_train/cunkuan_train/cunkuan_q3.csv', index=False)
        train_q4.to_csv(r'./x_train/cunkuan_train/cunkuan_q4.csv', index=False)
        test_q1.to_csv(r'./x_test/cunkuan_test/cunkuan_q1.csv', index=False)
        print("处理季度的客户信息")
        train_q3, train_q4, test_q1 = self.deal_cust()
        train_q3.to_csv(r'./x_train/cust_q3.csv', index=False)
        train_q4.to_csv(r'./x_train/cust_q4.csv', index=False)
        test_q1.to_csv(r'./x_train/cust_test.csv', index=False)
        
        
if __name__ == "__main__":
    dd = deal_data()
    dd.save_data()
        
        
        
        
        
                
                
        
