import sys,os
sys.path.append(str(os.getcwd())+"/竞赛")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(os.getcwd())+"/竞赛")
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics


class Data_Meger(object):
    def __init__(self):
        self._testA_path_ = '/home/mzlp/scikit_learn_data/hy_round1_testA_20200102'
        self._testB_path_ = '/home/mzlp/scikit_learn_data/hy_round1_testB_20200221'
        
        self._train_path_ = '/home/mzlp/scikit_learn_data/hy_round1_train_20200102'
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
        self.train_data = self.train_data.rename({'渔船ID':'id','速度':'v','方向':'d','time':'t'},axis=1)
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
        self.testA_data = self.testA_data.rename({'渔船ID':'id','速度':'v','方向':'d','time':'t'},axis=1)
        return self.testA_data.id.unique().size
    
    def generate_testB_data(self):
        files = os.listdir(self._testB_path_)
        temp = []
        for f in files:
            df = pd.read_csv(f"{self._testB_path_}/{f}")
            self._turn_time_(df)
            self._generate_slope_(df)
            self._generate_time_delta_(df)     
            temp.append(df)
        self.testB_data = pd.concat(temp)
        self.testB_data = self.testB_data.rename({'渔船ID':'id','速度':'v','方向':'d','time':'t'},axis=1)
        return self.testB_data.id.unique().size
    
    
    def generate_csv_files(self):
        if self.train_data is not None:
            self.train_data.to_csv('basic_data/train.csv',index=False,mode='w')
        if self.testA_data is not None:
            self.testA_data.to_csv('basic_data/testA.csv',index=False, mode='w')
        if self.testB_data is not None:
            self.testB_data.to_csv('basic_data/testB.csv',index=False, mode='w')
            
    
    def _turn_time_(self, data):
        temp = np.temp = np.zeros(data.index.stop) + 2019
        temp = temp.astype(str)
        temp = pd.Series(temp).str.slice(0, 4).str.cat(data['time'].astype(str))
        data['time'] = pd.to_datetime(temp)
        # 处理日期
        data['hour'] = data['time'].dt.hour
        data['weekday'] = data['time'].dt.weekday
        data['date'] = data['time'].dt.date
        data['date'] = pd.to_datetime(data['date'])
        data['time'] = pd.to_datetime(data['time'])
    
    # 生成slope
    def _generate_slope_(self,df):
        df['x_delta'] = df.x -df.x.shift(-1)
        df['y_delta'] = df.y -df.y.shift(-1)
        df['slope'] = df['x_delta'] / np.where(df['y_delta']==0,0.001,df['y_delta'])
        # df.drop('x_delta',1)
        # df.drop('y_delta',1)
    
    def _generate_time_delta_(self,df):
        df['time_delta'] = (df.time-df.time.shift(-1)).dt.total_seconds()
        df['v_delta'] = (df['速度']-df['速度'].shift(-1)).abs()
        # df['acc'] = (df['v_delta']/df['time_delta']).abs()
    
        
        


def load_data_from_csv():
    if not os.path.exists('train.csv'):
        return 'train.csv 不存在'
    if not os.path.exists('testB.csv'):
        return 'test.csv 不存在'
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('testB.csv')
    train_data.date = pd.to_datetime(train_data.date)
    train_data.t = pd.to_datetime(train_data.t)
    test_data.date = pd.to_datetime(test_data.date)
    test_data.t = pd.to_datetime(test_data.t)
    
    # train_data = train_data.rename(columns={'t':'time'})
    # test_data=test_data.rename(columns={'t':'time'})  
    
    return (train_data, test_data)

# load_data_from_csv()

# data = Data_Meger()
# data.generate_train_data()
# data.generate_test_data()
# data.generate_csv_files()
# train, test = load_data_from_csv()

def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


def add_v_0_big_count(train):
    v_0_count = train.groupby('id')['v'].agg({'v_0_count':lambda x:x[x==0].size}).reset_index()
    v_0_count_pro = train.groupby('id')['v'].agg({'v_0_count_pro':lambda x:x[x==0].size/x.size}).reset_index()
    v_big_count = train.groupby('id')['v'].agg({'v_big_count': lambda x:x[x>15].size}).reset_index()
    temp = pd.merge(v_0_count,v_big_count,on='id',how='left')
    temp = pd.merge(temp,v_0_count_pro, on='id', how='left')
    return temp

# 去除train中的异常ID
def clear_train(df,num=25):
    x_min, x_max = 5154884.262826221,7121988.280228527
    y_min, y_max = 4480409.373533728, 6778506.13403334
    df1 = df[df.x>=x_min][df.x<=x_max][df.y>=y_min][df.y<=y_max]
    print(x_min)
    a = df1.id.value_counts()
    b = a[a<num].reset_index()
    b.columns = ['id','count']
    df = df[~df.id.isin(b.id)]
    print(b.id)
    import gc
    del df1,a,b
    gc.collect()
    return df

# 添加x_delta与y_delta都为0时的数据条数比例
def generate_slope_feture(data):
    df = data.copy()
    a = df.groupby('id')['slope','x_delta','y_delta'].agg(lambda x:x[(x.x_delta==0)&(x.y_delta==0)]['slope'].size)['slope']
    b = df.id.value_counts().sort_index()
    slope_0_pro = (a/b).reset_index()
    slope_0_pro.columns=['id','slope_0_pro']
    import gc
    del df,a,b
    gc.collect()
    print(slope_0_pro.head())
    return slope_0_pro

# 添加v的v_less_pro, 然后大于等于0.3132，小于等于13的分成6个桶
def generate_v_feature(data, num=6):
    import gc
    df = data.copy()
    count_all = df.groupby('id')['x'].count().reset_index()
    a = ((df[(df.v<0.3132) & (df.v>0)].groupby('id')['v'].count())/count_all.x).reset_index()       
    a.columns = ['id','v_less_pro']
    # 添加分桶统计, 比例
    temp = a
    df2 = df[(df.v>=0.3132)&(df.v<=13)]
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
    temp.fillna(0,inplace=True)
    return temp    

# 添加v的分桶统计后的 mean  std统计值
def generate_v_2_feature(data,num=6, aggs=['mean','std']):
    import gc
    df = data.copy()
    df2 = df[(df.v>=0.3132)&(df.v<=13)]
    # df2.v = np.log10(df2.v+1)
    df2['v_cut_6'] = pd.cut(df2.v,num,labels=range(num))
    df3 = df2.groupby(['id','v_cut_6'])['v'].agg(aggs).unstack()
    columns = []
    for i in df3.columns:
        columns.append('v_'+i[0]+'_'+str(i[1]))
    
    df3.columns = columns
    df3 = df3.reset_index()
    df3.fillna(0,inplace=True)
    return df3    


def generate_feature_v_d(data):
    df = data[['id','x','y','d','v','t']].copy()
    v_d_max = df.groupby('id')[['y','x','d','v']].agg({'y':'max','x':'max','d':'max','v':'max'}).reset_index()
    v_d_max.columns = ['id','y_max','x_max','d_max','v_max']
    df = pd.merge(df, v_d_max[['id','d_max']], on='id', how='left')
    df = pd.merge(df, v_d_max[['id','v_max']], on='id', how='left')
    df = pd.merge(df, v_d_max[['id','x_max']], on='id', how='left')
    df = pd.merge(df, v_d_max[['id','y_max']], on='id', how='left')
    
    # d
    d_min_t = df[df.d_max==df.d][['id','d','t']].groupby('id')[['d','t']].agg({'t':'min'}).reset_index()
    d_min_t.columns = ['id','d_max','d_min_t']
    d_min_t['d_min_t']=pd.to_datetime(d_min_t.d_min_t)
    d_min_t['d_min_t']=d_min_t['d_min_t'].dt.hour
    d_min_t = d_min_t.drop('d_max',axis=1)
    
    
    d_max_t = df[df.d_max==df.d][['id','d','t']].groupby('id')[['d','t']].agg({'t':'max'}).reset_index()
    d_max_t.columns = ['id','d_max','d_max_t']
    d_max_t['d_max_t']=pd.to_datetime(d_max_t.d_max_t)
    d_max_t['d_max_t']=d_max_t['d_max_t'].dt.hour
    d_max_t = d_max_t.drop('d_max',axis=1)
    
    # v
    v_min_t = df[df.v_max==df.v][['id','v','t']].groupby('id')[['v','t']].agg({'t':'min'}).reset_index()
    v_min_t.columns = ['id','v_max','v_min_t']
    # print(v_min_t.head())
    v_min_t['v_min_t']=pd.to_datetime(v_min_t.v_min_t)
    v_min_t['v_min_t']=v_min_t['v_min_t'].dt.hour
    v_min_t = v_min_t.drop('v_max',axis=1)
    
    
    v_max_t = df[df.v_max==df.v][['id','v','t']].groupby('id')[['v','t']].agg({'t':'max'}).reset_index()
    v_max_t.columns = ['id','v_max','v_max_t']
    v_max_t['v_max_t']=pd.to_datetime(v_max_t.v_max_t)
    v_max_t['v_max_t']=v_max_t['v_max_t'].dt.hour
    v_max_t = v_max_t.drop('v_max',axis=1)
    
    #x_t 最大的x 对应的最大y
    x_y_max = df[df.x_max==df.x][['id','x','y']].groupby('id')[['x','y']].agg({'y':'max'}).reset_index()
    x_y_max.columns = ['id','x_max','x_y_max_2']
    x_y_max = x_y_max.drop('x_max',axis=1)
    
    # x_y_max.drop('x_max',axis=1)
    #x_time 最大的y 对应的最大x
    y_x_max = df[df.y_max==df.y][['id','x','y']].groupby('id')[['x','y']].agg({'x':'max'}).reset_index()
    y_x_max.columns = ['id','y_x_max_2','y_max']
    y_x_max = y_x_max.drop('y_max',axis=1)
    
    # y_x_max.drop('y_max',axis=1)


    
    a = pd.concat([d_min_t,d_max_t,v_min_t,v_max_t,x_y_max,y_x_max],axis=1)
    b = (a.T).drop_duplicates().T
    b['id'] = b.id.astype('int')
    print(b.shape)
    # b = b.drop(['x_max','y_max'],axis=1)
    return b



def extract_feature(train, aggs=['max','min','mean','std','skew','sum'],num=25):
    # x_min, x_max = 5147691.697622943,7123330.111503685
    # y_min, y_max = 4508207.026892569, 6653071.101800047
    # 根据test中得出的异常值
    # x_min, x_max = 5154884.262826221,7121988.280228527
    # y_min, y_max = 4480409.373533728, 6778506.13403334
    df = train.copy()
    # if 'type' in df.columns:
    #     print('train_data')
    #     df = clear_train(df,num)
    #     print(df.id.value_counts())
    
    train = group_feature(df, 'id', 'x', ['count'])
    # x_y = df[df.x>=x_min][df.x<=x_max][df.y>=y_min][df.y<=y_max].copy()
    from scipy import stats
    # x_y['x'] = pd.to_numeric(stats.boxcox(x_y.x,lmbda=4))
    # x_y['y'] = pd.to_numeric(stats.boxcox(x_y.y,lmbda=4))
  
    # x_y.x = np.log10(x_y.x)
    # x_y.y = np.log10(x_y.y)
    t = group_feature(df, 'id','x',aggs)
    train = pd.merge(train, t, on='id', how='left')
    t = group_feature(df, 'id','y',aggs)
    train = pd.merge(train, t, on='id', how='left')
    # 添加slope斜率的计算
    t = group_feature(df, 'id','slope',aggs)
    train = pd.merge(train, t, on='id', how='left')
    # 去除v的异常值
    train = pd.merge(train,add_v_0_big_count(df),on='id',how='left')
    # print(train.columns)
    # t = group_feature(df[df.v >=0.3132][df.v <= 13], 'id','v',aggs)
    t = group_feature(df, 'id','v',aggs)
    train = pd.merge(train, t, on='id', how='left')
    t = group_feature(df, 'id','d',aggs)
    train = pd.merge(train, t, on='id', how='left')
    
    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['slope_Max'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']
    
    mode_hour = df.groupby('id')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['id'].map(mode_hour)
    
    t = group_feature(df, 'id','hour',['max','min'])
    train = pd.merge(train, t, on='id', how='left')
    # train['hour_nunique'] = df.groupby('id')['hour'].nunique()
    # train['date_nunique'] = df.groupby('id')['date'].nunique()
    # hour_nunique = df.groupby('id')['hour'].nunique().reset_index()
    # train['hour_nunique'] = hour_nunique.hour
    # date_unique = df.groupby('id')['date'].nunique().reset_index()
    # train['date_nunique'] = date_unique.date 

    t = df.groupby('id')['t'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    print('---------')
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    t = t.drop('diff_time',1)
    train = pd.merge(train, t, on='id', how='left')
    # 添加slope_0_pro比例
    slope_0_pro = generate_slope_feture(df)
    # slope_0_pro = generate_slope_feture(df)
    
    train = pd.merge(train, slope_0_pro, on='id', how='left')
    # 添加v分组比例信息
    v_0_cut = generate_v_feature(df,6)
    train = pd.merge(train, v_0_cut, on='id', how='left')
    # 添加v的分组统计指标 mean, std
    v_2_feature = generate_v_2_feature(df,num=6, aggs=['mean','std','skew','median','sum'])
    train = pd.merge(train, v_2_feature, on='id', how='left')
    # 来自合作伙伴
    v_d_time = generate_feature_v_d(df)
    train = pd.merge(train, v_d_time, on='id', how='left')   
    train.fillna(0,inplace=True)
    
    if 'type' in df.columns:
        a = df[['id','type']].copy()
        a = a.drop_duplicates()
        train = pd.merge(train, a, on='id', how='left')
        train['type'] = train['type'].replace(['拖网','围网','刺网'],[0,1,2])
        train.to_csv('basic_data/train_feature.csv',mode='w',index=False)
    else:
        train.to_csv('basic_data/test_feature.csv',mode='w',index=False)
    return train

# =================================================
# 以下为第二版本的特征工程， 来源公开代码的

def extract_feature_v2(data):
    # x的特征
    feature_x = group_feature(data,'id','x',['max','mean','min'])
    feature_x_q_25 = data.groupby('id')['x'].quantile(0.25).reset_index()
    feature_x_q_25.columns = ['id','x_25']
    # y的特征
    feature_y = group_feature(data,'id','y',['max','mean','min'])
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
    
    
    
    