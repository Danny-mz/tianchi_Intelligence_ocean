import sys,os
sys.path.append(str(os.getcwd()))
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(os.getcwd()))
import xgboost as xgb
# import catboost as cab
from sklearn.model_selection import StratifiedKFold, train_test_split, validation_curve
from sklearn import metrics
from feature_engineering import Data_Meger,extract_feature_v2,generate_slope_feature,generate_v_feature_v2,generate_acc,generate_x_y_50_v_25_t
# from feature_engineering import load_data_from_csv
from feature_engineering import plot_feature_importance,generate_slope_feature,generate_acc
# import lgbm_model_v2 as lgbm2
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
#step 1 训练模型 =========================================
print('=======================')
print('step1, 抽取训练集特征')
#  训练数据
data = Data_Meger()
data.generate_train_data()
# data.generate_csv_files()
train_data = data.train_data


# train_data, test_data = load_data_from_csv()
train_feature = extract_feature_v2(train_data)
# 1.增加特征 y_max_x_min x_max_y_min slope_0_pro
slope_0_pro_train = generate_slope_feature(train_data)
print(slope_0_pro_train.head())
train_feature = pd.merge(train_feature,slope_0_pro_train,on='id',how='left')

train_feature['y_max_x_min'] = train_feature['y_max'] - train_feature['x_min']
train_feature['x_max_y_min'] = train_feature['x_max'] - train_feature['y_min']

# 2. 增加 v < 0.05的比例: v_less_pro,  增加 0.05<v<=13分 六个桶，每个桶的比例
from feature_engineering import generate_v_feature_v2
train_v_cut = generate_v_feature_v2(train_data)
train_feature = pd.merge(train_feature,train_v_cut,on='id',how='left')

# 3. 增加ACCmean max
# acc_train = generate_acc(train_data)
# train_feature = pd.merge(train_feature, acc_train, on='id',how='left')
# train_feature.fillna(0,inplace=True)
# 4. 删除 v_5_pro
# train_feature = train_feature.drop(['v_5_pro'],1)



kind=train_feature.type
features = [i for i in train_feature.columns if i not in ['id','type']]
# lgb模型

params={'num_leaves':20
        ,'max_depth':6
        ,'learning_rate':0.39
        ,'n_estimators':160
        # ,'class_weight':{0:2.5,1:3,2:5}
        ,'class_weight':{0:2.5,1:2.3,2:6.2}
        ,'objective':'multiclass'
        ,'n_jobs':-1
        ,'reg_alpha':0.2
        ,'reg_lambda':0}
llf=lgb.LGBMClassifier(**params)
xlf=xgb.XGBClassifier(max_depth=6
                      ,learning_rate=0.39
                      ,n_estimators=150
                      ,reg_alpha=0.04
                      ,n_jobs=-1
                      ,reg_lambda=0.2
                    ,importance_type='total_cover')
# 特征交叉
feature_llf = ['x_max', 'x_min', 'x_25', 'y_max', 'y_min', 'y_75', 'a', 'v_std','d', 'xy_cov', 'slope_0_pro', 'y_max_x_min', 'x_max_y_min','v_less_pro', 'v_0_pro', 'v_1_pro', 'v_2_pro', 'v_3_pro','v_4_pro']
feature_xlf = ['x_max', 'x_mean', 'x_min', 'x_25', 'y_max', 'y_mean', 'y_min','y_75', 'a', 'v_std', 'v_75', 'y_max_x_min', 'x_max_y_min','v_0_pro', 'v_1_pro', 'v_5_pro']
from sklearn.preprocessing import PolynomialFeatures
train_feature_llf = PolynomialFeatures(include_bias=False).fit_transform(train_feature[feature_llf])
train_feature_xlf = PolynomialFeatures(include_bias=False).fit_transform(train_feature[feature_xlf])
train_feature_llf = pd.DataFrame(train_feature_llf)
train_feature_xlf = pd.DataFrame(train_feature_xlf)

selector = SelectFromModel(llf,threshold=-np.inf,max_features=100).fit(train_feature_llf,kind)
feature_flag_llf = selector.get_support()
feature_llf_new = np.array(train_feature_llf.columns)[feature_flag_llf]

selector_xgb = SelectFromModel(xlf,threshold=-np.inf,max_features=46).fit(train_feature_xlf,kind)
feature_flag_xlf = selector_xgb.get_support()
feature_xlf_new = np.array(train_feature_xlf.columns)[feature_flag_xlf]

train_feature_llf = train_feature_llf[feature_llf_new]
train_feature_xlf = train_feature_xlf[feature_xlf_new]

# step 2 训练模型============================
print('=======================')
print('step2, 训练模型')
details=[]

data_train = train_feature[features]
# 存储每折的模型 权重
models = dict()
# print(test_data.columns)
# feature_llf = features
# feature_xlf = features
sk=StratifiedKFold(n_splits=20,shuffle=True,random_state=2020)
for index,(train,test) in enumerate(sk.split(data_train,kind)):
    
    # x_train_llf=data_train[feature_llf].iloc[train]
    x_train_llf = train_feature_llf.iloc[train]
    # x_test_llf=data_train[feature_llf].iloc[test]
    x_test_llf = train_feature_llf.iloc[test]
    
    x_train_xlf = train_feature_xlf.iloc[train]
    x_test_xlf = train_feature_xlf.iloc[test]
    # x_train_xlf = data_train[features].iloc[train]
    # x_test_xlf = data_train[features].iloc[test]
    
    y_train=kind.iloc[train]
    y_test=kind.iloc[test]
    
    # xlf.fit(x_train_xlf,y_train)
    xlf.fit(x_train_xlf,y_train, sample_weight=y_train.map({0:2.5,1:2.3,2:6.2}))

    pred_xgb=xlf.predict(x_test_xlf)
    weight_xgb=metrics.f1_score(y_test,pred_xgb,average='macro')
    
    llf.fit(x_train_llf,y_train)
    pred_llf=llf.predict(x_test_llf)
    weight_lgb=metrics.f1_score(y_test,pred_llf,average='macro')
    
    prob_xgb=xlf.predict_proba(x_test_xlf)
    prob_lgb=llf.predict_proba(x_test_llf)
#     prob_cab=clf.predict_proba(x_test)
    
    scores=[]
    ijk=[]
    weight=np.arange(0,1.05,0.1)
    for i,item1 in enumerate(weight):
        prob_end=prob_xgb*item1+prob_lgb*(1-item1)
        score=metrics.f1_score(y_test,np.argmax(prob_end,axis=1),average='macro')
        scores.append(score)
        ijk.append((item1,1-item1))
#   存储权重 最大得分权重
    ii=ijk[np.argmax(scores)][0]
    jj=ijk[np.argmax(scores)][1]
#     kk=ijk[np.argmax(scores)][2]
    
    details.append(max(scores))
    details.append(weight_xgb)
    details.append(weight_lgb)
#     details.append(weight_cab)
    details.append(ii)
    details.append(jj)
    models.update({index:[xlf,llf,weight_xgb,weight_lgb]})
#     details.append(kk)


    print(index,' : ',max(scores))
    
df=pd.DataFrame(np.array(details).reshape(int(len(details)/5),5)
                ,columns=['test_end_score','xgboost','lightgbm'
                ,'weight_xgboost','weight_lightgbm'])
print(df.sort_values('test_end_score'))
print(df.mean())
print('=====开始预测========')
# step 3 =================================
print('=======================')
print('step3, 预测数据抽取')
# 预测数据抽取
data.generate_testA_data()
test_data = data.testA_data
test_feature = extract_feature_v2(test_data)
slope_0_pro_test = generate_slope_feature(test_data)
test_feature = pd.merge(test_feature,slope_0_pro_test,on='id',how='left')
test_feature['y_max_x_min'] = test_feature['y_max'] - test_feature['x_min']
test_feature['x_max_y_min'] = test_feature['x_max'] - test_feature['y_min']
test_v_cut = generate_v_feature_v2(test_data)
test_feature = pd.merge(test_feature,test_v_cut,on='id',how='left')

# 3. 增加ACCmean max
# acc_test = generate_acc(test_data)
# test_feature = pd.merge(test_feature, acc_test, on='id',how='left')
# test_feature.fillna(0,inplace=True)
# 4. 删除 v_5_pro
# test_feature = test_feature.drop(['v_5_pro'],1)
# 特征交叉
test_feature_llf = PolynomialFeatures(include_bias=False).fit_transform(test_feature[feature_llf])
test_feature_xlf = PolynomialFeatures(include_bias=False).fit_transform(test_feature[feature_xlf])
test_feature_llf = pd.DataFrame(test_feature_llf)
test_feature_xlf = pd.DataFrame(test_feature_xlf)

test_feature_llf = test_feature_llf[feature_llf_new]
test_feature_xlf = test_feature_xlf[feature_xlf_new]

data_test = test_feature[features]
#step 4 预测数据 ========================
# 预测模型
print('=======================')
print('step4, 预测数据')
answers=[]
for index in range(len(models)):
    xgb_model = models.get(index)[0]
    lgb_model = models.get(index)[1]
    xgb_weight = models.get(index)[2]
    lgb_weight = models.get(index)[3]
    
    test_xgb=xgb_model.predict_proba(test_feature_xlf)
    test_lgb=lgb_model.predict_proba(test_feature_llf)
    
    ans=test_xgb*xgb_weight + test_lgb*lgb_weight
    
    answers.append(np.argmax(ans,axis=1))
    answers.append(xgb_model.predict(test_feature_xlf))
    answers.append(lgb_model.predict(test_feature_llf))


print(np.array(answers).shape)
fina=[]
for i in range(len(test_feature.id)):
    counts=np.bincount(np.array(answers,dtype='int')[:,i])
    fina.append(np.argmax(counts))
end=pd.DataFrame(test_feature.id,columns=['id'])
end["type"]=pd.Series(fina).map({0:'拖网',1:'围网',2:'刺网'})

print('精度:\n',df.mean())
print('分布比例:\n',end.type.value_counts(1))
print('测试集数目:', len(test_feature.id))
end.to_csv('result.csv',mode='w',header=False, index=False,encoding='utf-8')

# lgbm2.generate_csv_file_from_yTestA(end,df.mean()[0])
# file = lgbm2.generate_csv_file_from_yTestA(end,df.mean()[0])
# a = lgbm2.compare_file('20200221_105112_0.9193B.csv', file)
# print('不一致数目:',a)