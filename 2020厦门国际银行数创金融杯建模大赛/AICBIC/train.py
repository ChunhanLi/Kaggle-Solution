# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:39:37 2020

@author: 31058
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy,ks_2samp
import gc
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
# 数据路径
os.chdir(r'D:/competition/2020_10_30/给选手数据')
class train_model():
    def __init__(self, left,
                       right,
                       test,
                       learning_rate=0.1,
                       seed=2020,
                       max_depth=10,
                       num_leaves=128,
                       **kwargs):
        """
        初始化超参数以及初始化各季度数据
        """
        self.params = {
            'learning_rate': learning_rate,
            'metric': 'multi_error',
            'objective': 'multiclass',
            'num_class': 3,
            'feature_fraction': 0.80,
            'bagging_fraction': 0.75,
            'bagging_freq': 2,
            'n_jobs': 4,
            'seed': seed,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
        }
        self.params.update(kwargs)
        self.left = left
        self.right = right
        self.test = test

    def kappa(self, preds, y_true):
        """
        作用：kappa评价函数
        """
        preds = np.argmax(preds, axis=1)
        score = cohen_kappa_score(y_true, preds)
        
        return 'kappa',score,True
    
    def search_weight(self, valid_y, raw_prob, class_num, step=0.001):
        """
        作用：结果后处理函数
        """
        weight=[1.0]*class_num
        f_best = cohen_kappa_score(valid_y, raw_prob.argmax(axis=1))
        flag_score = 0
        round_num = 1
        while(flag_score != f_best):
            print("round: ", round_num)
            round_num += 1
            flag_score = f_best
            for c in range(class_num):
                for n_w in range(0, 2000,10):
                    num = n_w * step
                    new_weight = weight.copy()
                    new_weight[c] = num
    
                    prob_df = raw_prob.copy()
                    prob_df = prob_df * np.array(new_weight)
    
                    f = cohen_kappa_score(valid_y, prob_df.argmax(
                        axis=1))
                    if f > f_best:
                        weight = new_weight.copy()
                        f_best = f
                        print(f)
                        
        return weight
    
    def feature(self):
        """
        作用：二次加工特征
        """
        # 月底账户金额合计
        self.left['aum_all_3'] = self.left['X1_3'] + self.left['X2_3'] + self.left['X3_3'] + self.left['X4_3'] + self.left['X5_3'] + self.left['X6_3'] \
        + self.left['X7_3'] + self.left['X8_3']
        self.right['aum_all_3'] = self.right['X1_3'] + self.right['X2_3'] + self.right['X3_3'] + self.right['X4_3'] + self.right['X5_3'] \
        + self.right['X6_3'] + self.right['X7_3'] + self.right['X8_3']
        self.test['aum_all_3'] = self.test['X1_3'] + self.test['X2_3'] + self.test['X3_3'] + self.test['X4_3'] + self.test['X5_3'] \
        + self.test['X6_3'] + self.test['X7_3'] + self.test['X8_3']

        for col in ['last_label', 'last_sum']:
            self.left[col] = np.nan
        # 拼接上期合计账户金额，构造窗口特征
        self.right = self.right.merge(self.left[['cust_no', 'label', 'aum_all_3']].rename(columns={'label':'last_label', 
                                                                                                       'aum_all_3':'last_sum',
                                                                                                       }), how='left', on=['cust_no'])
        self.test = self.test.merge(self.right[['cust_no', 'label', 'aum_all_3']].rename(columns={'label':'last_label', 
                                                                                                    'aum_all_3':'last_sum',
                                                                                                    }), how='left', on=['cust_no'])
        self.right['last_sum'] = self.right['aum_all_3']/self.right['last_sum']
        self.test['last_sum'] = self.test['aum_all_3']/self.test['last_sum']
        train = pd.concat([self.left, self.right])
        train = train.reset_index(drop=True)
        for i in range(1,4):
            # 各个月除贷款外金额合计
            train['aum_sum_%s'%i] = train['X1_%s'%i] + train['X2_%s'%i] +train['X3_%s'%i] +train['X4_%s'%i] +train['X5_%s'%i] +train['X6_%s'%i] +train['X8_%s'%i]
            # 活期存款余额占比
            train['X3per_%s'%i] = train['X3_%s'%i]/train['aum_sum_%s'%i]
            # 账户金额与个人年收入比例
            train['aumper_%s'%i] = train['aum_sum_%s'%i]/train['I11']
            train['aum_all_%s'%i] = train['X1_%s'%i] + train['X2_%s'%i] +train['X3_%s'%i] +train['X4_%s'%i] +train['X5_%s'%i] +train['X6_%s'%i] +train['X7_%s'%i] +train['X8_%s'%i]
            
            self.test['aum_sum_%s'%i] = self.test['X1_%s'%i] + self.test['X2_%s'%i] + self.test['X3_%s'%i] + self.test['X4_%s'%i] + \
            self.test['X5_%s'%i] + self.test['X6_%s'%i] + self.test['X8_%s'%i]
            self.test['X3per_%s'%i] = self.test['X3_%s'%i]/self.test['aum_sum_%s'%i]
            self.test['aumper_%s'%i] = self.test['aum_sum_%s'%i]/self.test['I11']
            self.test['aum_all_%s'%i] = self.test['X1_%s'%i] + self.test['X2_%s'%i] + self.test['X3_%s'%i] + self.test['X4_%s'%i] \
            + self.test['X5_%s'%i] + self.test['X6_%s'%i] + self.test['X7_%s'%i] + self.test['X8_%s'%i]
                
        change_fea = ['aum_all']
        for col in change_fea:
            # 逐月变化比
            train['%s32_change'%col] = (train['%s_3'%col]-train['%s_2'%col])/train['%s_2'%col]
            train['%s21_change'%col] = (train['%s_2'%col]-train['%s_1'%col])/train['%s_1'%col]
            train['%s31_change'%col] = (train['%s_3'%col]-train['%s_1'%col])/train['%s_1'%col]
            self.test['%s32_change'%col] = (self.test['%s_3'%col]-self.test['%s_2'%col])/self.test['%s_2'%col]
            self.test['%s21_change'%col] = (self.test['%s_2'%col]-self.test['%s_1'%col])/self.test['%s_1'%col]
            self.test['%s31_change'%col] = (self.test['%s_3'%col]-self.test['%s_1'%col])/self.test['%s_1'%col]
        # 删除可能出现共线性特征
        train = train.drop(columns=['%s_1'%n for n in change_fea]+['%s_2'%n for n in change_fea])
        self.test = self.test.drop(columns=['%s_1'%n for n in change_fea]+['%s_2'%n for n in change_fea]) 
        # 一致性检验
        col_dict = []
        for col in list(train.columns):
            if col in ['label', 'cust_no']:
                continue
            ans = ks_2samp(train[col].notna().values, self.test[col].notna().values)
            if ans.pvalue<0.00000001:
                col_dict.append(col)
        print("一致性检验未通过特征有：", ','.join(col_dict))
        # 删除一致性检验交叉且泛化效果不好特征
        train_x = train.drop(columns=['cust_no', 'label', 'I9']+['B6_min', 'B6_max', 'B6_mean', 'B6_median', 'I5', 'B6_std', 'B6_skew'])
        train_y = train['label']+1
        test_x = self.test.drop(columns=['cust_no', 'I9']+['B6_min', 'B6_max', 'B6_mean', 'B6_median', 'I5', 'B6_std', 'B6_skew'])  
        
        return train_x, train_y, test_x
    
    def train_model(self, train_x, train_y, test_x):
        """
        作用：训练模型
        ------params------
        train_x:训练特征
        train_y:训练标签
        test_x:测试特征
        ------return------
        result:测试集结果包含cust_no与label两列
        result_pb:测试集概率文件，结果融合使用，包含cust_no,negative_one,zero,one四列
        """
        scores = []
        imp = pd.DataFrame()
        imp['feat'] = train_x.columns
        
        oof_train = np.zeros((len(train_x), 3))
        preds = np.zeros((len(test_x), 3))
        folds = 5
        seeds = [44]#, 2020, 527, 1527]
        for seed in seeds:
            kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
                x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[val_idx]
                train_set = lgb.Dataset(x_trn, y_trn)
                val_set = lgb.Dataset(x_val, y_val)
        
                model = lgb.train(self.params, train_set, num_boost_round=500000,
                                  valid_sets=(train_set, val_set), early_stopping_rounds=100,
                                  verbose_eval=20)
                oof_train[val_idx] += model.predict(x_val) / len(seeds)
                preds += model.predict(test_x) / folds / len(seeds)
                scores.append(model.best_score['valid_1']['multi_error'])
                imp['gain' + str(fold + 1)] = model.feature_importance(importance_type='gain')
                imp['split' + str(fold + 1)] = model.feature_importance(importance_type='split')
        print("线下CV的kappa为", self.kappa(oof_train, train_y))
        weight = self.search_weight(train_y, oof_train, class_num=3, step=0.001)
        preds_new = preds*np.array(weight)
        self.test['negative_one'] = preds_new[:,0]
        self.test['zero'] = preds_new[:,1]
        self.test['one'] = preds_new[:,2]
        preds_new = preds_new.argmax(axis=1)-1
        self.test['label'] = preds_new
        result = self.test[['cust_no', 'label']]
        result_pb = self.test[['cust_no', 'negative_one', 'zero', 'one']]
        
        return result,result_pb
    
    def main(self):
        train_x, train_y, test_x = self.feature()
        result, result_pb = self.train_model(train_x, train_y, test_x)
        result.to_csv(r'result1208.csv', index=False)
        result_pb.to_csv(r'result1208_p.csv', index=False)
        
        
if __name__ == "__main__":
    # 数据读取
    train_aum_q3 = pd.read_csv(r'./x_train/aum_train/train_aum_q3_concat.csv')
    train_aum_q4 = pd.read_csv(r'./x_train/aum_train/train_aum_q4_concat.csv')
    train_behavior_q3 = pd.read_csv(r'./x_train/behavior_train/train_behavior_q3.csv')
    train_behavior_q4 = pd.read_csv(r'./x_train/behavior_train/train_behavior_q4.csv')
    test_aum_q1 = pd.read_csv(r'./x_test/aum_test/test_aum_q1_concat.csv')
    test_behavior_q1 = pd.read_csv(r'./x_test/behavior_test/test_behavior_q1.csv')
    train_be_q3 = pd.read_csv(r'./x_train/big_event_train/train_be_q3.csv')
    train_be_q4 = pd.read_csv(r'./x_train/big_event_train/train_be_q4.csv')
    test_be_q1 = pd.read_csv(r'./x_test/big_event_test/test_be_q4.csv')
    train_ck_q3 = pd.read_csv(r'./x_train/cunkuan_train/cunkuan_q3.csv')
    train_ck_q4 = pd.read_csv(r'./x_train/cunkuan_train/cunkuan_q4.csv')
    test_ck_q1 = pd.read_csv(r'./x_test/cunkuan_test/cunkuan_q1.csv')
    train_cust_q3 = pd.read_csv(r'./x_train/cust_q3.csv')
    train_cust_q4 = pd.read_csv(r'./x_train/cust_q4.csv')
    test_cust = pd.read_csv(r'./x_train/cust_test.csv')
    left = train_aum_q3.merge(train_behavior_q3.drop(columns=['label']),
                                                          on='cust_no').merge(train_be_q3.drop(columns=['label']),
                                                          how='left',
                                                          on='cust_no').merge(train_ck_q3.drop(columns=['label']),
                                                          how='left',
                                                          on='cust_no').merge(train_cust_q3,
                                                          how='left',
                                                              on='cust_no')   
    right = train_aum_q4.merge(train_behavior_q4.drop(columns=['label']),
                                                          how='left',
                                                          on='cust_no').merge(train_be_q4.drop(columns=['label']),
                                                          how='left',
                                                          on='cust_no').merge(train_ck_q4.drop(columns=['label']),
                                                          how='left',
                                                          on='cust_no').merge(train_cust_q4,
                                                          how='left',
                                                          on='cust_no')   
    test = test_aum_q1.merge(test_behavior_q1,
                                                          how='left',
                                                          on='cust_no').merge(test_be_q1,
                                                          how='left',
                                                          on='cust_no').merge(test_ck_q1,
                                                          how='left',
                                                          on='cust_no').merge(test_cust,
                                                          how='left',
                                                          on='cust_no') 

    train_process = train_model(left, right, test)   
    train_process.main()
        
    
    