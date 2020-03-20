import sys,os
sys.path.append(str(os.getcwd()))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(os.getcwd()))


class Data_Meger(object):
    def __init__(self):
        # self._testA_path_ = '/tcdata/hy_round2_testA_20200225'
        self._testA_path_ = '/tcdata/hy_round2_testB_20200312'
        # self._testA_path_ = '/home/mzlp/scikit_learn_data/hy_round1_testB_20200221'
        
        self._train_path_ = '/tcdata/hy_round2_train_20200225'
        # self._train_path_ = '/home/mzlp/scikit_learn_data/hy_round2_train_20200225'
        
        self.train_data = None
        self.testA_data = None
        self.testB_data = None
    
    def generate_train_data(self):
        files = os.listdir(self._train_path_)
        temp = []
        for f in files:
            df = pd.read_csv(f"{self._train_path_}/{f}")
            self._turn_time_(df)
            self._generate_slope_(df)
            self._generate_time_delta_(df)
            temp.append(df)
        self.train_data = pd.concat(temp)
        self.train_data = self.train_data.rename({'渔船ID':'id','lat':'x','lon':'y','速度':'v','方向':'d','time':'t'},axis=1)
        print(self.train_data.columns)
        return self.train_data['id'].unique().size
    
    def generate_testA_data(self):
        files = os.listdir(self._testA_path_)
        temp = []
        for f in files:
            df = pd.read_csv(f"{self._testA_path_}/{f}")
            self._turn_time_(df)
            self._generate_slope_(df)
            self._generate_time_delta_(df)     
            temp.append(df)
        self.testA_data = pd.concat(temp)
        self.testA_data = self.testA_data.rename({'渔船ID':'id','lat':'x','lon':'y','速度':'v','方向':'d','time':'t'},axis=1)
        return self.testA_data.id.unique().size
    
    def generate_csv_files(self):
        if self.train_data is not None:
            self.train_data.to_csv('train_B.csv',index=False,mode='w')
        if self.testA_data is not None:
            self.testA_data.to_csv('testA.csv',index=False, mode='w')
        # if self.testB_data is not None:
        #     self.testB_data.to_csv('basic_data/testB.csv',index=False, mode='w')
            
    
    def _turn_time_(self, data):
        temp = np.temp = np.zeros(data.index.stop) + 2019
        temp = temp.astype(str)
        temp = pd.Series(temp).str.slice(0, 4).str.cat(data['time'].astype(str))
        data['time'] = pd.to_datetime(temp)
        # 处理日期
        # data['hour'] = data['time'].dt.hour
        # data['weekday'] = data['time'].dt.weekday
        # data['date'] = data['time'].dt.date
        # data['date'] = pd.to_datetime(data['date'])
        data['time'] = pd.to_datetime(data['time'])
    
    # 生成slope
    def _generate_slope_(self,df):
        df['x_delta'] = df.lat -df.lat.shift(-1)
        df['y_delta'] = df.lon -df.lon.shift(-1)
        df['slope'] = df['x_delta'] / np.where(df['y_delta']==0,0.001,df['y_delta'])
        # df.drop('x_delta',1)
        # df.drop('y_delta',1)
    
    def _generate_time_delta_(self,df):
        df['time_delta'] = (df.time-df.time.shift(-1)).dt.total_seconds()
        df['v_delta'] = (df['速度']-df['速度'].shift(-1)).abs()
        # df['acc'] = (df['v_delta']/df['time_delta']).abs()


# =================================================
# 以下为第二版本的特征工程， 来源公开代码的

def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

# 1. 基础特征
def extract_feature_v2(data):
    # x的特征
    feature_x = group_feature(data,'id','x',['max','mean','min','std'])
    feature_x_q_25 = data.groupby('id')['x'].quantile(0.25).reset_index()
    feature_x_q_25.columns = ['id','x_25']
    # y的特征
    feature_y = group_feature(data,'id','y',['max','mean','min','std'])
    feature_y_q_75 = data.groupby('id')['y'].quantile(0.75).reset_index()    
    feature_y_q_75.columns = ['id','y_75']
    # x y 协方差
    xy_cov = data[['id','x','y']].groupby('id').agg(lambda x:x.x.cov(x.y))['x'].reset_index()
    xy_cov.columns = ['id','xy_cov']
    # a
    x_mean = data.groupby('id').apply(lambda x: (x.x.diff().iloc[1:].abs()/x.t.diff().iloc[1:].dt.total_seconds()).mean())
    y_mean = data.groupby('id').apply(lambda y: (y.y.diff().iloc[1:].abs()/y.t.diff().iloc[1:].dt.total_seconds()).mean())
    a = np.sqrt(x_mean**2 + y_mean**2).reset_index()
    a.columns = ['id','a']
    # v的特征
    v_mean_std = group_feature(data,'id','v',['mean','std'])
    v_q_75 = data.groupby('id')['v'].quantile(0.75).reset_index()
    v_q_75.columns = ['id','v_75']
    # 方向 d
    d_mean = data.groupby('id')['d'].mean().reset_index()
    data_feature = None
    # type
    if 'type' in data.columns.tolist():
        d_type = data.groupby('id')['type'].agg(lambda x:x.iloc[0]).reset_index()
        data_feature = pd.concat([feature_x,feature_x_q_25,feature_y,feature_y_q_75,a,v_mean_std,v_q_75,d_mean,xy_cov,d_type],axis=1)
        data_feature['type'] = data_feature['type'].replace({'拖网':0,'围网':1,'刺网':2})
        data_feature = data_feature.T.drop_duplicates().T 
        data_feature['type'] = data_feature.type.astype('int')
        
    else:
        data_feature = pd.concat([feature_x,feature_x_q_25,feature_y,feature_y_q_75,a,v_mean_std,v_q_75,d_mean,xy_cov],axis=1)
        data_feature = data_feature.T.drop_duplicates().T
    data_feature['id'] = data_feature.id.astype('int')
    return data_feature
  
# 2. 添加x_delta与y_delta都为0时的数据条数比例
def generate_slope_feature(data):
    df = data.copy()
    a = df.groupby('id')['slope','x_delta','y_delta'].agg(lambda x:x[(x.x_delta==0)&(x.y_delta==0)]['slope'].size)['slope']
    b = df.id.value_counts().sort_index()
    slope_0_pro = (a/b).reset_index()
    slope_0_pro.columns=['id','slope_0_pro']
    import gc
    print(slope_0_pro.head())
    del df,a,b
    gc.collect()
    return slope_0_pro
    
# 3. 增加 v < 0.05的比例: v_less_pro,  增加 0.05<v<=13分 六个桶，每个桶的比例
def generate_v_feature_v2(data, num=6):
    import gc
    df = data.copy()
    count_all = df.groupby('id')['x'].count().reset_index()
    v_less_pro = ((df[(df.v<=0.05) & (df.v>=0)].groupby('id')['v'].count())).reset_index()      
    v_less_pro.columns = ['id','v_less_pro']
    temp = pd.merge(count_all,v_less_pro, on='id',how='left')
    temp['v_less_pro'] = temp['v_less_pro']/temp['x']
    temp = temp[['id','v_less_pro']]
    # v_less_pro=v_less_pro.fillna(0,inplace=True)
    # 添加分桶统计, 比例
    # temp = v_less_pro
    df2 = df[(df.v>0.05)&(df.v<=13)]
    # df2.v = np.log10(df2.v+1)
    df3 = df2.groupby('id')['v'].count().reset_index()
    df2['v_cut_6'] = pd.cut(df2.v,num,labels=range(num))
    b = df2.groupby(['id','v_cut_6'])['v_cut_6'].count().unstack()
    for i in range(num):
        m = []
        m = b[i].reset_index()
        m[i] = m[i] / df3.v
        m.columns=['id','v_{}_pro'.format(str(i))]
        temp = pd.merge(temp, m, on='id',how='left')
    
    # c = df2[['id','v','v_cut_6']].groupby(['id','v_cut_6']).agg(['mean','std','skew']).unstack()
    # for i in c.items():
    #     m = i[1].reset_index()
    #     m.columns = ['id',i[0][0]+'_'+str(i[0][2])+'_'+i[0][1]]
    #     temp = pd.merge(temp, m, on='id',how='left')
    
    temp.fillna(0,inplace=True)
    print(temp.head())
    return temp  

# 4. 增加 acc_mean acc_max
def generate_acc(data):
    df = data[(data.v>0.05)&(data.v<=13)].copy()
    acc_mean = df.groupby('id').apply(lambda x:(x.v.diff().iloc[1:]/(x.t.diff().iloc[1:].dt.total_seconds()/3600)).abs().quantile(0.5))
    acc_mean = acc_mean.reset_index()
    acc_mean.columns = ['id','acc_mean']
    acc_max = df.groupby('id').apply(lambda x:(x.v.diff().iloc[1:]/(x.t.diff().iloc[1:].dt.total_seconds()/3600)).abs().quantile(0.99))
    acc_max = acc_max.reset_index()
    acc_max.columns = ['id','acc_max']
    acc_25 = df.groupby('id').apply(lambda x:(x.v.diff().iloc[1:]/(x.t.diff().iloc[1:].dt.total_seconds()/3600)).abs().quantile(0.25))
    acc_25 = acc_25.reset_index()
    acc_25.columns = ['id','acc_25']
    acc_75 = df.groupby('id').apply(lambda x:(x.v.diff().iloc[1:]/(x.t.diff().iloc[1:].dt.total_seconds()/3600)).abs().quantile(0.75))
    acc_75 = acc_75.reset_index()
    acc_75.columns = ['id','acc_75']
    temp = pd.merge(acc_mean, acc_max, on='id',how='left')
    temp = pd.merge(temp, acc_25, on='id',how='left')
    temp = pd.merge(temp, acc_75, on='id',how='left')
    temp.fillna(0,inplace=True)
    print(temp.head())
    return temp

# 5. 增加 x_50 y_50 v_25  time_delta_mean
def generate_x_y_50_v_25_t(data):
    t_delta_mean = data.groupby('id').apply(lambda x:(x.t.diff().iloc[1:].dt.total_seconds()/60).abs().mean()).reset_index()
    t_delta_mean.columns = ['id','t_delta_mean']
    v_25 = data.groupby('id').apply(lambda x:x.v.quantile(0.25)).reset_index()
    v_25.columns = ['id','v_25']
    x_50 = data.groupby('id').apply(lambda x:x.x.quantile(0.5)).reset_index()
    x_50.columns = ['id','x_50']
    y_50 = data.groupby('id').apply(lambda x:x.y.quantile(0.5)).reset_index()
    y_50.columns = ['id','y_50']
    temp = pd.merge(t_delta_mean,v_25,on='id',how='left')
    temp = pd.merge(temp,x_50,on='id',how='left')
    temp = pd.merge(temp,y_50,on='id',how='left')
    print(temp.head())
    return temp 
    
    

# 绘制特征重要性图
def plot_feature_importance(data,features,model):
    width=0.25
    address=np.arange(len(data[features].columns))
    tick_label=data[features].columns
    feature_importances = model.feature_importances_/sum(model.feature_importances_)
    plt.figure(dpi=600,figsize=(3.6,1.5))
    plt.bar(address
            ,feature_importances
            ,width
            ,color='c'
            ,label='lightgbm')
    plt.legend(fontsize=4, loc='upper left')
    plt.axhline(np.mean(feature_importances),alpha=0.2,linewidth=0.6, linestyle='--',color='red')
    plt.text(-1.5,np.mean(feature_importances),str(np.mean(feature_importances).round(2)),size=4)
    plt.axhline(np.quantile(feature_importances,0.2),alpha=0.2,linewidth=0.6, linestyle='--',color='red')
    plt.xticks(address
                ,tick_label
                ,fontsize=4
                ,rotation=90
                ,horizontalalignment='center')
    plt.yticks(fontsize=4,rotation=0)
    plt.title('lightgbm——feature_importances_',fontsize=4)
    plt.show()
    
    
    
    