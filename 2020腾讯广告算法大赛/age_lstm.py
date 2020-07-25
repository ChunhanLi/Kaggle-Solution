import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit,StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold,train_test_split,GroupShuffleSplit,StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error,log_loss,confusion_matrix,accuracy_score
import sqlite3

import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
#from bayes_opt import BayesianOptimization
import re
from string import punctuation
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from tqdm.notebook import tqdm
#from numba import jit
from collections import Counter
import json
import joblib
import multiprocessing
import time
import keras
from sklearn.preprocessing import StandardScaler,OneHotEncoder
#from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras import models
from keras import layers
from keras.models import *
from keras.layers import *
import tensorflow.keras as tk
import math
from tensorflow.keras.layers import Attention

from keras import initializers
from keras.callbacks import *
from keras.optimizers import *

import keras
from keras.utils import multi_gpu_model

df_w2v_creative_id = pd.read_pickle('../w2v_file_set_version_v3/w2v_creative_id_embedding_200-50-5-10-set-version.pkl')
df_w2v_ad_id = pd.read_pickle('../w2v_file_set_version_v3/w2v_ad_id_embedding_200-50-5-10-set-version.pkl')
df_w2v_advertiser_id = pd.read_pickle('../w2v_file_set_version_v3/w2v_advertiser_id_embedding_128-50-5-10-set-version.pkl')
df_w2v_product_id = pd.read_pickle('../w2v_file_set_version_v3/w2v_product_id_embedding_128-50-5-10-set-version.pkl')
#df_w2v_industry_id = pd.read_pickle('w2v_file_list_version/w2v_industry_embedding_32-10-3-5-list-version.pkl')
print('creative_id',df_w2v_creative_id.shape)
print('ad_id',df_w2v_ad_id.shape)
print('advertiser_id',df_w2v_advertiser_id.shape)
print('product_id',df_w2v_product_id.shape)
#print('industry_id',df_w2v_industry_id.shape)




embed_cre=pd.read_pickle('../fit_generator_set_v3/embed_cre.pkl').values

embed_ad=pd.read_pickle('../fit_generator_set_v3/embed_ad.pkl').values

embed_adv=pd.read_pickle('../fit_generator_set_v3/embed_adv.pkl').values

embed_prod=pd.read_pickle('../fit_generator_set_v3/embed_prod.pkl').values

#embed_industry=pd.read_pickle('./fit_generator_list/embed_industry.pkl').values
#df_bag = pd.read_pickle('../fit_generator_set_v3/df_bag_generator.pkl')[:3000000]
tokenizer_cre = Tokenizer(lower=False, char_level=False, split=',')
tokenizer_ad = Tokenizer(lower=False, char_level=False, split=',')
tokenizer_adv = Tokenizer(lower=False, char_level=False, split=',')
tokenizer_prod = Tokenizer(lower=False, char_level=False, split=',')

tokenizer_cre.fit_on_texts(df_w2v_creative_id['creative_id'])
tokenizer_ad.fit_on_texts(df_w2v_ad_id['ad_id'])
tokenizer_adv.fit_on_texts(df_w2v_advertiser_id['advertiser_id'])
tokenizer_prod.fit_on_texts(df_w2v_product_id['product_id'])

del df_w2v_creative_id,df_w2v_ad_id,df_w2v_advertiser_id,df_w2v_product_id
gc.collect()

df_bag = pd.read_pickle('../fit_generator_set_v3/df_bag_generator.pkl')[:3000000][['list_creative_id','list_ad_id','list_advertiser_id','list_product_id']]

embed_cre = embed_cre.astype(np.float32)
embed_ad = embed_ad.astype(np.float32)
embed_adv = embed_adv.astype(np.float32)
embed_prod = embed_prod.astype(np.float32)


adam = keras.optimizers.adam(learning_rate = 0.003)
def model_lstm(embed_cre,embed_ad,embed_adv,embed_prod):

    K.clear_session()
    # The embedding layer containing the word vectors
    with tf.device('/cpu:0'):
        emb_layer_cr = Embedding(
            input_dim=embed_cre.shape[0],
            output_dim=embed_cre.shape[1],
            weights=[embed_cre],
            input_length=100,
            trainable=False
        )
        emb_layer_ad = Embedding(
            input_dim=embed_ad.shape[0],
            output_dim=embed_ad.shape[1],
            weights=[embed_ad],
            input_length=100,
            trainable=False
        )
        emb_layer_adv = Embedding(
            input_dim=embed_adv.shape[0],
            output_dim=embed_adv.shape[1],
            weights=[embed_adv],
            input_length=50,
            trainable=False
        )
        emb_layer_pro = Embedding(
            input_dim=embed_prod.shape[0],
            output_dim=embed_prod.shape[1],
            weights=[embed_prod],
            input_length=50,
            trainable=False
        )
    #     emb_layer_ind = Embedding(
    #         input_dim=embed_industry.shape[0],
    #         output_dim=embed_industry.shape[1],
    #         weights=[embed_industry],
    #         input_length=20,
    #         trainable=False
    #     )

    
    
    lstm_layer_cr = Bidirectional(
            LSTM(400, recurrent_dropout=0.1, dropout=0.25, return_sequences=True))
    lstm_layer_ad = Bidirectional(
            LSTM(400, recurrent_dropout=0.1, dropout=0.25, return_sequences=True))
    lstm_layer_adv = Bidirectional(
            LSTM(400, recurrent_dropout=0.1, dropout=0.25, return_sequences=True))
    lstm_layer_prod = Bidirectional(
            LSTM(200, recurrent_dropout=0.1, dropout=0.25, return_sequences=True))
#     lstm_layer_ind = Bidirectional(
#             LSTM(16, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
    # 1D convolutions that can iterate over the word vectors
    conv1_cr = Conv1D(filters=400, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_ad = Conv1D(filters=400, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_adv = Conv1D(filters=400, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_prod = Conv1D(filters=400, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_ind = Conv1D(filters=200, kernel_size=1,
                   padding='same', activation='relu',)

    seq_cr = Input(shape=(100,))
    seq_ad = Input(shape=(100,))
    seq_adv = Input(shape=(50,))
    seq_prod = Input(shape=(50,))

    #seq_ind = Input(shape=(20,))


    emb_cr = emb_layer_cr(seq_cr)
    emb_ad = emb_layer_ad(seq_ad)
    emb_adv = emb_layer_adv(seq_adv)
    emb_prod = emb_layer_pro(seq_prod)
    #emb_ind = emb_layer_ind(seq_ind)
    
    lstm_cr = lstm_layer_cr(emb_cr)
    lstm_ad = lstm_layer_ad(emb_ad)
    lstm_adv = lstm_layer_adv(emb_adv)
    lstm_prod = lstm_layer_prod(emb_prod)

    
    
    conv1a_cr = conv1_cr(lstm_cr)
    conv1a_ad = conv1_ad(lstm_ad)
    conv1a_adv = conv1_adv(lstm_adv)
    conv1a_prod = conv1_prod(lstm_prod)
    
    # Run through CONV + GAP layers
    gap1a_cr = GlobalAveragePooling1D()(conv1a_cr)
    gmp1a_cr = GlobalMaxPool1D()(conv1a_cr)

    gap1a_ad = GlobalAveragePooling1D()(conv1a_ad)
    gmp1a_ad = GlobalMaxPool1D()(conv1a_ad)
    
    gap1a_adv = GlobalAveragePooling1D()(conv1a_adv)
    gmp1a_adv = GlobalMaxPool1D()(conv1a_adv)
    
    gap1a_prod = GlobalAveragePooling1D()(conv1a_prod)
    gmp1a_prod = GlobalMaxPool1D()(conv1a_prod)
    
    #gap1a_ind = GlobalAveragePooling1D()(conv1a_ind)
    #gmp1a_ind = GlobalMaxPool1D()(conv1a_ind)
    
    merge1 = concatenate([gap1a_cr, gmp1a_cr,gap1a_ad,gmp1a_ad,\
                         gap1a_adv,gmp1a_adv,gap1a_prod,gmp1a_prod])

    
    x = Dropout(0.3)(merge1)
    x = BatchNormalization()(x)
    x = Dense(1500, activation='relu',)(x)
    x = Dropout(0.45)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu',)(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu',)(x)
    x = Dropout(0.35)(x)
    x = BatchNormalization()(x)
    pred = Dense(10, activation='softmax')(x)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Model(inputs=[seq_cr,seq_ad,seq_adv,seq_prod], outputs=pred)
        model_gpu2=multi_gpu_model(model, gpus=4)
        model_gpu2.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model_gpu2

def augmention(list1):
    list1 = list(list1)
    n = len(list1)
    seed = np.random.rand(1)[0]
    if seed > 0.9:
        sample_size = max(int(n*0.4),1)
    elif seed > 0.8:
        sample_size = max(int(n*0.8),1)
    elif seed > 0.7:
        sample_size = max(int(n*0.9),1)
    elif seed > 0.6:
        sample_size = max(int(n*0.95),1)
    else:
        sample_size = n
    return list(np.random.permutation(list1)[:sample_size])

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, data_X,data_Y, tok_list,max_lens,batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.data_X = data_X
        self.data_Y = data_Y
        self.indexes = np.arange(len(self.data_X))
        self.shuffle = shuffle
        self.tok_list = tok_list
        self.max_lens = max_lens
        self.on_epoch_end()

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data_X) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取data_X集合中的数据
        batch_data_X = self.data_X.iloc[batch_indexs,:]
        batch_data_Y = self.data_Y[batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_data_X,batch_data_Y)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data_X,batch_data_Y):
        X_train_list = []
        targets = ['creative_id','ad_id','advertiser_id','product_id']
        for i,target in enumerate(targets):
            shuffled_target = batch_data_X['list_'+target].map(lambda x:augmention(x))
            shuffled_target = self.tok_list[i].texts_to_sequences(shuffled_target)
            shuffled_target = pad_sequences(shuffled_target,maxlen = self.max_lens[i],value = 0)
            X_train_list.append(shuffled_target)
            #X_train_list.append(1-shuffled_target.clip(max=1))

        #如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return X_train_list, batch_data_Y

class DataGenerator_pred(keras.utils.Sequence):
    
    def __init__(self, data_X,data_Y, tok_list,max_lens,batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.data_X = data_X
        self.data_Y = data_Y
        self.indexes = np.arange(len(self.data_X))
        self.shuffle = shuffle
        self.tok_list = tok_list
        self.max_lens = max_lens
        self.on_epoch_end()

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data_X) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取data_X集合中的数据
        batch_data_X = self.data_X.iloc[batch_indexs,:]
        batch_data_Y = self.data_Y[batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_data_X,batch_data_Y)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data_X,batch_data_Y):
        X_train_list = []
        targets = ['creative_id','ad_id','advertiser_id','product_id']
        for i,target in enumerate(targets):
            shuffled_target = batch_data_X['list_'+target].map(lambda x:list(np.random.permutation(x)))
            shuffled_target = self.tok_list[i].texts_to_sequences(shuffled_target)
            shuffled_target = pad_sequences(shuffled_target,maxlen = self.max_lens[i],value = 0)
            X_train_list.append(shuffled_target)
            #X_train_list.append(1-shuffled_target.clip(max=1))

        #如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return X_train_list, batch_data_Y

user =  pd.read_pickle('../w2v_file_set_version_v3/user.pkl')
user.age = user.age.astype(np.int16)
user.gender = user.gender.astype(np.int8)

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=33)
#user =  pd.read_pickle('../raw_data/processed_data/user.pkl')
#X_train = user.iloc[:,3:]
#print(X_train.shape)
y_train_age = user['age'] - 1
#y_train_gender = user['gender'] - 1
from keras.utils.np_utils import to_categorical
y_train_age = to_categorical(y_train_age, num_classes=10)
fea_impor = 0
k = 1
pred_gender = 0
y_train_pred = np.zeros(y_train_age.shape)
y_train_pred_prob = np.zeros((y_train_age.shape[0],10))
#y_train_pred_prob = np.zeros((y_train_gender.shape[0],1))
score_list = []
for train_index,test_index in kf.split(df_bag[:3000000],user['age']):
    print(f'fold_{k}*********************************************')
    file_name = f'model_v12_fold{k}_set_tiaocan.hdf5'
    k+=1
    if k <=1:
        continue
    X_train2 = df_bag[:3000000].iloc[train_index,:]
    y_train2 = y_train_age[train_index]
    
    #X_test2 = df_bag.iloc[test_index,:]
    X_test2 = df_bag[:3000000].iloc[test_index,:]
    y_test2 = y_train_age[test_index]

    #model_checkpoint = ModelCheckpoint(file_name,save_best_only=True, verbose=1, monitor='val_accuracy', mode='auto')
    class ParallelModelCheckpoint(ModelCheckpoint):
        def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=1):
            self.single_model = model
            super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

        def set_model(self, model):
            super(ParallelModelCheckpoint,self).set_model(self.single_model)
    model = model_lstm(embed_cre,embed_ad,embed_adv,embed_prod)
    model_checkpoint = ParallelModelCheckpoint(model, filepath=file_name, monitor='val_accuracy',  save_weights_only=False, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=2, min_lr=0.0015, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, verbose=10,restore_best_weights=True)
    train_generator = DataGenerator(X_train2,y_train2,\
                    tok_list = [tokenizer_cre,tokenizer_ad,tokenizer_adv,tokenizer_prod],\
                    max_lens = [100,100,50,50],batch_size = 3000,shuffle=True)
    valid_generator = DataGenerator_pred(X_test2,y_test2,\
                    tok_list = [tokenizer_cre,tokenizer_ad,tokenizer_adv,tokenizer_prod],\
                    max_lens = [100,100,50,50],batch_size = 10000,shuffle=False)
    model.fit_generator(train_generator,\
              validation_data  = valid_generator, 
               epochs=200,verbose=1, callbacks=[es,reduce_lr,model_checkpoint],workers = 16,use_multiprocessing=True)
    break
#     y_train_pred_prob[test_index,:] = model.predict([X_test_cre2,X_test_ad2,X_test_adv2,X_test_prod2])
#     pred_gender+= model.predict([X_test_cre,X_test_ad,X_test_adv,X_test_prod])/kf.n_splits




#### pred

class DataGenerator_pred(keras.utils.Sequence):
    
    def __init__(self, data_X, tok_list,max_lens,batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.data_X = data_X
        #self.data_Y = data_Y
        self.indexes = np.arange(len(self.data_X))
        self.shuffle = shuffle
        self.tok_list = tok_list
        self.max_lens = max_lens
        self.on_epoch_end()

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data_X) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取data_X集合中的数据
        batch_data_X = self.data_X.iloc[batch_indexs,:]
        #batch_data_Y = self.data_Y[batch_indexs]

        # 生成数据
        X= self.data_generation(batch_data_X)

        return X

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data_X):
        X_train_list = []
        targets = ['creative_id','ad_id','advertiser_id','product_id']
        for i,target in enumerate(targets):
            shuffled_target = batch_data_X['list_'+target].map(lambda x:list(np.random.permutation(x)))
            shuffled_target = self.tok_list[i].texts_to_sequences(shuffled_target)
            shuffled_target = pad_sequences(shuffled_target,maxlen = self.max_lens[i],value = 0)
            X_train_list.append(shuffled_target)
        #如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return X_train_list


import keras
from keras.utils import multi_gpu_model

from keras.models import load_model
train_generator = DataGenerator_pred(df_bag[3000000:],
                tok_list = [tokenizer_cre,tokenizer_ad,tokenizer_adv,tokenizer_prod],\
                max_lens = [100,100,50,50],batch_size = 5000,shuffle=False)
all1 = 0
for k in range(1,3):
    file_name =  f'model_v12_fold{k}_set_final.hdf5'
    model = load_model(file_name)
    for _ in range(5):
        print(k,_)
        tmp_pred = model.predict(train_generator,verbose=1,workers = 32,use_multiprocessing=True)
        all1 += tmp_pred/25
        gc.collect()

#### oof

from keras.models import load_model
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=33)
user =  pd.read_pickle('../w2v_file_set_version_v3/user.pkl')
#X_train = user.iloc[:,3:]
#print(X_train.shape)
y_train_age = user['age'] - 1
y_train_gender = user['gender'] - 1
from keras.utils.np_utils import to_categorical
y_train_age = to_categorical(y_train_age, num_classes=10)
k = 1
pred_gender = 0
y_train_pred = np.zeros(y_train_age.shape[0])
y_train_pred_prob = np.zeros((y_train_age.shape[0],10))
#y_train_pred_prob = np.zeros((y_train_gender.shape[0],1))
score_list = []
for train_index,test_index in kf.split(df_bag[:3000000],user['age']):
    

    print(f'fold_{k}*********************************************')
    file_name = f'model_v12_fold{k}_set_final.hdf5'
    k+=1
    if k>=4:
        break
    X_train2 = df_bag[:3000000].iloc[train_index,:]
    y_train2 = y_train_age[train_index]
    
    #X_test2 = df_bag.iloc[test_index,:]

    y_test2 = y_train_age[test_index]
    tmp_pred = 0
    
    model = load_model(file_name)
    for __ in range(5):
        train_generator = DataGenerator_pred(df_bag.loc[test_index,:],
                    tok_list = [tokenizer_cre,tokenizer_ad,tokenizer_adv,tokenizer_prod],\
                    max_lens = [100,100,50,50],batch_size = 5000,shuffle=False)
        tmp111 = model.predict(train_generator,verbose=1,workers = 8,use_multiprocessing=True)/5
        print(accuracy_score(user['age'][test_index],tmp111.argmax(axis=1)+1))
        tmp_pred += tmp111
    
    y_train_pred_prob[test_index] = tmp_pred
    print(f'Fold_{k-1}',accuracy_score(user['age'][test_index],tmp_pred.argmax(axis=1)+1))
    score_list.append(accuracy_score(user['age'][test_index],tmp_pred.argmax(axis=1)+1))
    #pred_gender += (model.predict([X_test_cre,X_test_ad,X_test_adv,X_test_prod],batch_size = 4096)) /kf.n_splits
print(np.mean(score_list))
# pd.DataFrame(y_train_pred_prob).to_pickle('v29_age_oof_prob.pkl')
# pd.DataFrame(pred_gender).to_pickle('v29_age_pred_prob.pkl')