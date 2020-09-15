import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest,chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

df=pd.read_csv("heart.csv")
x=df.iloc[:,0:13]
y=df.iloc[:,13]
feature_train,feature_test,label_train,label_test=train_test_split(x,y,test_size=0.2)

dt=DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2)
dt=dt.fit(feature_train,label_train)
pred=dt.predict(feature_test)
accuracy=accuracy_score(label_test,pred)
print(accuracy)

rf=RandomForestClassifier(n_estimators=10,max_depth=40)
rf=rf.fit(feature_train,label_train)
pred=rf.predict(feature_test)
accuracy=accuracy_score(label_test,pred)
print(accuracy)



nb=GaussianNB()
nb=nb.fit(feature_train,label_train)
pred=nb.predict(feature_test)
accuracy=accuracy_score(label_test,pred)
print(accuracy)



ada=AdaBoostClassifier(n_estimators=10,base_estimator=dt,learning_rate=1)
ada=ada.fit(feature_train,label_train)
pred=ada.predict(feature_test)
accuracy=accuracy_score(label_test,pred)
print(accuracy)



from sklearn.svm import SVC
svc=SVC(kernel='linear',gamma=100)
svc.fit(feature_train,label_train)
pred=svc.predict(feature_test)
accuracy=accuracy_score(label_test,pred)
print(accuracy)


import pickle as p
p.dump(rf,open('final_heartdisease_model.pickle','wb'))
