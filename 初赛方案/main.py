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
from feature_engineering import Data_Meger, load_data_from_csv,extract_feature, clear_train,generate_v_feature,generate_feature_v_d, group_feature,extract_feature_v2
from feature_engineering import plot_feature_importance,generate_slope_feture
import lgbm_model_v2 as lgbm2
from sklearn.feature_selection import SelectFromModel

train_data, test_data = load_data_from_csv()
train_feature = extract_feature_v2(train_data)
test_feature = extract_feature_v2(test_data)
# 1.增加特征 y_max_x_min x_max_y_min slope_0_pro
slope_0_pro_train = generate_slope_feture(train_data)
slope_0_pro_test = generate_slope_feture(test_data)
train_feature = pd.merge(train_feature,slope_0_pro_train,on='id',how='left')
test_feature = pd.merge(test_feature,slope_0_pro_test,on='id',how='left')

train_feature['y_max_x_min'] = train_feature['y_max'] - train_feature['x_min']
train_feature['x_max_y_min'] = train_feature['x_max'] - train_feature['y_min']
test_feature['y_max_x_min'] = test_feature['y_max'] - test_feature['x_min']
test_feature['x_max_y_min'] = test_feature['x_max'] - test_feature['y_min']
# 2. 增加 v < 0.05的比例: v_less_pro,  增加 0.05<v<=13分 六个桶，每个桶的比例
from feature_engineering import generate_v_feature_v2
train_v_cut = generate_v_feature_v2(train_data)
test_v_cut = generate_v_feature_v2(test_data)
train_feature = pd.merge(train_feature,train_v_cut,on='id',how='left')
test_feature = pd.merge(test_feature,test_v_cut,on='id',how='left')

kind=train_feature.type
features = [i for i in train_feature.columns if i not in ['id','type']]
# lgb模型
params={'num_leaves':20
        ,'max_depth':6
        ,'learning_rate':0.39
        ,'n_estimators':160
        ,'class_weight':{0:2.5,1:3,2:5}
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
feature_llf = ['x_max', 'x_min', 'x_25', 'y_max', 'y_min', 'y_75', 'a', 'v_std',
       'd', 'xy_cov', 'slope_0_pro', 'y_max_x_min', 'x_max_y_min',
       'v_less_pro', 'v_0_pro', 'v_1_pro', 'v_2_pro', 'v_3_pro',
       'v_4_pro']
feature_xlf = ['x_max', 'x_mean', 'x_min', 'x_25', 'y_max', 'y_mean', 'y_min',
       'y_75', 'a', 'v_std', 'v_75', 'y_max_x_min', 'x_max_y_min',
       'v_0_pro', 'v_1_pro', 'v_5_pro']

details=[]
answers=[]
data_train = train_feature[features]
data_test = test_feature[features]
# print(test_data.columns)

sk=StratifiedKFold(n_splits=20,shuffle=True,random_state=2020)
for index,(train,test) in enumerate(sk.split(data_train,kind)):
    
    x_train_llf=data_train[feature_llf].iloc[train]
    x_test_llf=data_train[feature_llf].iloc[test]
    
    x_train_xlf = data_train[feature_xlf].iloc[train]
    x_test_xlf = data_train[feature_xlf].iloc[test]
    
    y_train=kind.iloc[train]
    y_test=kind.iloc[test]
    
    xlf.fit(x_train_xlf,y_train)
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
#     details.append(kk)

    print(index,' : ',max(scores))
   
    test_xgb=xlf.predict_proba(data_test[feature_xlf])
    test_lgb=llf.predict_proba(data_test[feature_llf])
#     test_cab=clf.predict_proba(test_data)
    ans=test_xgb*ii+test_lgb*jj
    
    answers.append(np.argmax(ans,axis=1))
    answers.append(xlf.predict(data_test[feature_xlf]))
    answers.append(llf.predict(data_test[feature_llf]))

df=pd.DataFrame(np.array(details).reshape(int(len(details)/5),5)
                ,columns=['test_end_score','xgboost','lightgbm'
                ,'weight_xgboost','weight_lightgbm'])

fina=[]
for i in range(2000):
    counts=np.bincount(np.array(answers,dtype='int')[:,i])
    fina.append(np.argmax(counts))
end=pd.DataFrame(np.arange(9000,11000,1),columns=['id'])
end["type"]=pd.Series(fina).map({0:'拖网',1:'围网',2:'刺网'})
file = lgbm2.generate_csv_file_from_yTestB(end,df.mean()[0])
file
a = lgbm2.compare_file('20200221_105112_0.9193B.csv', file)
print('不一致数目:',a)