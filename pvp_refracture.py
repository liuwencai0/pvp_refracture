import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler

#应用标题
st.title('A machine learning-based predictive model for predicting Refracture of Percutaneous vertebroplasty')

# conf
st.sidebar.markdown('## Variables')
#Age = st.sidebar.slider("Age", 1, 99, value=25, step=1)
#high = st.sidebar.slider("High(cm)", 100, 200, value=168, step=1)
BMI = st.sidebar.slider("BMI", 10.0, 50.0, value=25.0, step=0.1)
BMD = st.sidebar.slider("BMD", 1.0, 10.0, value=5.0, step=0.1)
#hospitalization_time = st.sidebar.slider("Hospitalization time", 1, 60, value=10, step=1)
#surgery_time = st.sidebar.slider("Surgery_time(min)", 15, 150, value=60, step=1)
#Injury_to_surgery = st.sidebar.slider("Injury to surgery", 1, 400, value=50, step=1)
#Laterality = st.sidebar.selectbox('Laterality',('Left','Right','Other'),index=0)
#Race = st.sidebar.selectbox("Race",('White','Black','Other'))
Multiple = st.sidebar.selectbox("Multiple vertebral fracture",('No','Yes'))
Antiosteoporosis = st.sidebar.selectbox("Anti-osteoporosis therapy",('No','Yes'))
Steroid = st.sidebar.selectbox("Steroid use",('No','Yes'))
#Laterality = st.sidebar.selectbox("Laterality",('Left','Right','Not a paired site'))
#Grade = st.sidebar.selectbox("Grade",('Well differentiated','Moderately differentiated','Poorly differentiated'
#                                      ,'Undifferentiated; anaplastic','Unknow'))
#T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4','TX'))
#N = st.sidebar.selectbox("N stage",('N0','N1','NX'))
#Liver_metastasis = st.sidebar.selectbox("Liver.metastasis",('No','Yes'),index=0)
#Lung_metastasis = st.sidebar.selectbox("Lung.metastasis",('No','Yes'),index=0)
#Chemotherapy = st.sidebar.selectbox("Chemotherapy",('No','Yes'),index=0)
#Sequence_number = st.sidebar.selectbox("Sequence.number",('One primary only','1st primaries','2nd primaries','more'))
#Lung_metastases = st.sidebar.selectbox("Lung metastases",('No','Yes'))

# str_to_int

map = {'No':0,'Yes':1}
#Age =map[Age]
#Laterality =map[Laterality]
#Sex =map[Sex]
Multiple =map[Multiple]
Antiosteoporosis =map[Antiosteoporosis]
Steroid =map[Steroid]
#Grade =map[Grade]
#T =map[T]
#N =map[N]
#surgery =map[surgery]
#Radiation =map[Radiation]
#Sequence_number =map[Sequence_number]
#Liver_metastasis =map[Liver_metastasis]
#Lung_metastasis =map[Lung_metastasis]



# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['refracture'] = thyroid_train['refracture'].apply(lambda x : +1 if x==1 else 0)
#thyroid_test = pd.read_csv('test.csv', low_memory=False)
#thyroid_test['refracture'] = thyroid_test['refracture'].apply(lambda x : +1 if x==1 else 0)

features = ['BMI','BMD','Antiosteoporosis','Multiple','Steroid']

#全部变量
#features = ['Age','Sex','high','weigh','BMI','BMD','hospitalization.time','Injection.volume.of.bone.cement','surgery.time','Hospital.stay.to.surgery','Injury.to.surgery','Antiosteoporosis','Multiple','refracture','Steroid']#
target = 'refracture'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

RF = RandomForestClassifier(n_estimators=21,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
RF.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (RF.predict_proba(np.array([[BMI,BMD,Antiosteoporosis,Multiple,Steroid]]))[0][1])> sp
prob = (RF.predict_proba(np.array([[BMI,BMD,Antiosteoporosis,Multiple,Steroid]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for Refracture:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of Refracture:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0.737 ;Accuracy: 0.784 ;Sensitivity(recall): 0.550 ;Specificity :0.839 ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')





