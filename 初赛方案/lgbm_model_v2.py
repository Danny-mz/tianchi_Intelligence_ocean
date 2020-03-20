import sys,os
sys.path.append(str(os.getcwd()))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics

from feature_engineering import Data_Meger, load_data_from_csv,extract_feature

# def LGBM_model(train_feature=None, test_feature=None, function_list=['max','min','mean','std','skew','sum'], params=None,num=25,features=None, crosFeatures=False):
def LGBM_model(train_feature=None, test_feature=None, params=None,features=None):
    # if train is None or test is None:
        # global train,test 
        # train, test = load_data_from_csv()
    
    # train_data = extract_feature((train), function_list,num=num)
    # test_data = extract_feature(test,function_list)
    # print(train_data.head(5))
    # print(test_data.head(5))
    if features is None:
        features = [x for x in train_feature.columns if x not in ['id','type','t','diff_time','date','hour']]
    target = 'type'
    # 设置训练集合
    fold = StratifiedKFold(n_splits=7,shuffle=True, random_state=42)
    train_x = train_feature[features].copy()
    test_data = test_feature[features]
    
    train_y = train_feature[target]
    # train_y = train_y.replace(['拖网','围网','刺网'],[0,1,2])
    
    score_train = []
    score_test = []
    models = []
    pred_y = np.zeros((len(test_data),3))
    oof = np.zeros((len(train_x),3))
    if params is None:
        params = {'n_estimators': 5000, 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 3,'early_stopping_rounds': 150}
    
    for index ,(train_index, test_index) in enumerate(fold.split(train_x,train_y)):
        train_set = lgb.Dataset(train_x.iloc[train_index],train_y.iloc[train_index])
        val_set = lgb.Dataset(train_x.iloc[test_index], train_y.iloc[test_index])
        # model = 
        model = lgb.train(params,train_set, valid_sets=[train_set, val_set], verbose_eval=150)
        models.append(model)
        val_pred = model.predict(train_x.iloc[test_index])
        oof[test_index] = val_pred
        val_y = train_y.iloc[test_index]
        val_pred = np.argmax(val_pred, axis=1)
        print("=={}=================".format(index))
        f1_test =  metrics.f1_score(val_y,val_pred, average='macro')
        score_test.append(f1_test)
        print(index, '测试集：  f1: ',f1_test)
        
        train_y_pred = model.predict(train_x.iloc[train_index])
        f1_train = metrics.f1_score(train_y.iloc[train_index],np.argmax(train_y_pred,1), average='macro')
        score_train.append(f1_train)        
        print(index, '训练集：  f1: ', f1_train)
        
        print("ACC",metrics.accuracy_score(val_y, val_pred))
        print("======================================")
        temp_y = model.predict(test_data)
        pred_y = pred_y + temp_y / 7
    
    # 总体得分
    oof = np.argmax(oof,1)
    f1 = metrics.f1_score(train_y, oof, average='macro')
    acc = metrics.accuracy_score(train_y, oof)
    mat = metrics.confusion_matrix(train_y, oof)
    print('========总体得分==============')
    print("F1: ", f1)
    print("ACC: ", acc)
    print("mat:\n", mat)
    f1_test =  np.mean(score_test)
    f1_train =  np.mean(score_train)
    
    print('测试集：',f1_test)
    print('训练集：',f1_train)    
    print('======================')
    import gc
    # del train, test
    gc.collect()
    return (f1_test,f1_train,models)

def model_from_sk(train_feature, features, params,cv=7):
    # test_data = test_feature[features]
    if features is None:
        features = [x for x in train_feature.columns if x not in ['id','type']]
    # target = 'type'
    x = train_feature[features]
    y = train_feature['type']
    # y = y.replace(['拖网','围网','刺网'],[0,1,2])
    
    models = []
    scores_test = []
    scores_train = []
    # pred_y = np.zeros((len(test_data), 3))
    oof = np.zeros(len(x))
    
    fold = StratifiedKFold(n_splits=cv,shuffle=True,random_state=42)
    
    for index, (train_index, test_index) in enumerate(fold.split(x,y)):
        model = lgb.LGBMClassifier(**params)
        x_test = x.iloc[test_index,:]
        y_test = y.iloc[test_index]
        x_train = x.iloc[train_index,:]
        y_train = y.iloc[train_index]
        
        model.fit(x_train, y_train)
        models.append(model)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        oof[train_index] = train_pred
        # temp_y = model.predict_proba(test_data)
        # pred_y = pred_y + temp_y / cv
        
        f1_test = metrics.f1_score(y_test,test_pred,average='macro')
        f1_train = metrics.f1_score(y_train,train_pred,average='macro')
        scores_test.append(f1_test)
        scores_train.append(f1_train)
        print('======{}======='.format(index))
        print(index,' 训练集 F1: ', f1_train)
        print(index,' 测试集 F1: ', f1_test)
        print('----------')
        print(index,' 训练集 : ', model.score(x_train,y_train))
        print(index,' 测试集 : ', model.score(x_test,y_test))
        print('====结束=======')
        
    # 总体得分
    # oof = np.argmax(oof,1)
    f1 = metrics.f1_score(y, oof, average='macro')
    f1_test = np.mean(scores_test)
    f1_train = np.mean(scores_train)
    
    acc = metrics.accuracy_score(y, oof)
    mat = metrics.confusion_matrix(y, oof)
    print('========总体得分==============')
    print('测试集： ', f1_test)
    print('训练集： ', f1_train)
    print("F1: ", f1)
    print("ACC: ", acc)
    print("mat:\n", mat)
    print('======================')
    return (f1_test, models)
        
def predict_test(models, feature_test):
    y = np.zeros((len(feature_test),3))
    cv = len(models)
    for model in models:
        temp = model.predict_proba(feature_test)
        y = y + temp / cv
    
    result = np.argmax(y, 1)
    testA = pd.DataFrame({'id':range(7000,9000),'type':result}) 
    testA['type'].value_counts(1)
    return testA

def predict_testB(models, feature_test):
    y = np.zeros((len(feature_test),3))
    cv = len(models)
    for model in models:
        temp = model.predict_proba(feature_test)
        y = y + temp / cv
    
    result = np.argmax(y, 1)
    testb = pd.DataFrame({'id':range(9000,11000),'type':result}) 
    testb['type'].value_counts(1)
    return testb

def predict_test_v2(model, feature_test):
    # y = np.zeros((len(feature_test),2))
    # cv = len(models)
    # for model in models:
    #     temp = model.predict_proba(feature_test)
    #     y = y + temp / cv
    
    result = model.predict(feature_test)
    testA = pd.DataFrame({'id':range(7000,9000),'type':result}) 
    testA['type'].value_counts(1)
    return testA   
    
    


def model_validate_params(train_feature, test_feature,params):
    x_new = train_feature.iloc[:,:-1]
    y = train_feature.iloc[:,-1]
    x_test = test_feature
    models = []
    score=[]
    pred_y = np.zeros((len(x_test),3))
    oof = np.zeros((len(x_new),3))
    # params = {'n_estimators': 5000, 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 3,'early_stopping_rounds': 150,'learning_rate':0.07,'metric':'multi_logloss','lambda_l1':0.05,'min_data_in_leaf':20,'is_unbalance':True,'max_depth':3}
    fold = StratifiedKFold(n_splits=7,shuffle=True, random_state=42)
    for index ,(train_index, test_index) in enumerate(fold.split(x_new,y)):
        train_set = lgb.Dataset(x_new.iloc[train_index,:],y.iloc[train_index])
        val_set = lgb.Dataset(x_new.iloc[test_index,:], y.iloc[test_index])
        # model = lgb.train(params,train_set, valid_sets=[train_set, val_set], verbose_eval=150)
        model = lgb.train(params,train_set, valid_sets=[train_set,val_set], verbose_eval=150)
    
        models.append(model)
        val_pred = model.predict(x_new.iloc[test_index,:])
        oof[test_index] = val_pred
        val_y = y.iloc[test_index]
        val_pred = np.argmax(val_pred, axis=1)
        print("== {} 次====================================".format(index))
        f1_test = metrics.f1_score(val_y,val_pred, average='macro')
        f1_train = metrics.f1_score(y.iloc[train_index],np.argmax(model.predict(x_new.iloc[train_index,:]),axis=1), average='macro')
        
        score.append(f1_train)
        print(index, '测试集 f1: ', f1_test)
        print(index, '训练集集 f1: ', f1_train)
        
        print("\n ACC",metrics.accuracy_score(val_y, val_pred))
        print("======================================")
        temp_y = model.predict(x_test)
        pred_y = pred_y + temp_y / 7

   # 总体得分
    oof = np.argmax(oof,1)
    f1 = metrics.f1_score(y, oof, average='macro')
    acc = metrics.accuracy_score(y, oof)
    mat = metrics.confusion_matrix(y, oof)
    f1_train = np.mean(score)
    print('========总体得分==============')
    print("测试集 F1: ", f1)
    print("训练集 F1: ", f1_train)
    
    print("\nACC: ", acc)
    print("\nmat:\n", mat)
    print('======================')
    return (f1,f1_train)




# 根据特征重要性阈值抽取特征
def feature_extract(models,feature_importance=100):
    ret = []
    for index, model in enumerate(models):
        df = pd.DataFrame()
        df['name'] = model.feature_name()
        df['score'] = model.feature_importance()
        df['index'] = index
        ret.append(df)

    ret = pd.concat(ret)

    df = ret.groupby('name',as_index=False)['score'].mean()
    df = df.sort_values(['score'], ascending=False)

    df.to_csv('feature_importance.csv',index=False)
    feature_list=df[df.score>=feature_importance].name
    print(feature_list)
    return feature_list.to_list()


def generate_csv_file_from_yTestA(ytestA,f1_score):
    if len(ytestA) != 2000:
        return '请检查数据集，条目不对'
    import datetime
    date_str = str(datetime.datetime.now().date()).replace('-','')
    time_str = str(datetime.datetime.now().time()).replace(':','')[:6]
    name = date_str +"_"+ time_str+'_' + str(round(f1_score,4) ) +'.csv'
    ytestA['type'] = ytestA['type'].replace([0,1,2], ['拖网','围网','刺网'])
    ytestA.to_csv('data/'+name,mode='w',header=False, index=False,encoding='utf-8')
    
    # file = open('竞赛/data/'+name,'w',encoding='utf-8')
    return name


def generate_csv_file_from_yTestB(ytestB,f1_score,path=os.getcwd()):
    if len(ytestB) != 2000:
        return '请检查数据集，条目不对'
    import datetime
    date_str = str(datetime.datetime.now().date()).replace('-','')
    time_str = str(datetime.datetime.now().time()).replace(':','')[:6]
    name = date_str +"_"+ time_str+'_' + str(round(f1_score,4) )+'B' +'.csv'
    ytestB['type'] = ytestB['type'].replace([0,1,2], ['拖网','围网','刺网'])
    ytestB.to_csv(path+'/'+name,mode='w',header=False, index=False,encoding='utf-8')
    
    # file = open('竞赛/data/'+name,'w',encoding='utf-8')
    return name



# testA的特征数据
def compare_file(file1_name,file2_name):
    path = os.getcwd()+'/{}'
    file1 = path.format(file1_name)
    file2 = path.format(file2_name)
    pd1 = pd.read_csv(file1, header=None)
    pd2 = pd.read_csv(file2, header=None)
    pd1.columns=['id','csv_1']
    pd2.columns = ['id', 'csv_2']
    d1 = pd1['csv_1'].value_counts(1)
    d2 = pd2['csv_2'].value_counts(1)
    df = pd.merge(pd1, pd2)
    df['diff'] = df.iloc[:,1] == df.iloc[:,2]
    result = df[ df['diff'] == False ]
    a = pd1.groupby(['csv_1']).count()
    b = pd2.groupby(['csv_2']).count()
    c = pd.concat([a,b], axis=1)
    c.columns=['csv_1','csv_2']
    print(result)
    print('总共不一致数目：' + str(result['id'].count()))
    print(c)
    print(pd.concat([d1,d2],axis=1))
    return str(result['id'].count())

