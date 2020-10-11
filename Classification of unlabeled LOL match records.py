# Classification of unlabeled LoL match records with "Win/Loss"
import numpy as np
import pandas as pa
import time
from texttable import Texttable

from sklearn import metrics
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# data to train
# usecols: columns of classifiers to train the data, 
# column 0 not chosen for it's gameId,column 1 not chosen for it's creationTime, column 3 not chosen for it's seasonId
# column 4 is the winner
df1=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",
                usecols=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
df2=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",usecols=[4])
datasets=df1.values
labels=df2.values
l=np.ravel(labels)

# data to test
df1_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",
                usecols=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
df2_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",usecols=[4])
datasets_test=df1_test.values
labels_test=df2_test.values
l_test=np.ravel(labels_test)

# classifiers

DT=tree.DecisionTreeClassifier(max_depth=33,min_samples_split=100)  # Decision Tree
start = time.perf_counter()
DT.fit(datasets,l)
pred2=DT.predict(datasets_test)
end = time.perf_counter()
t1=[end-start]
a2=metrics.accuracy_score(labels_test,pred2)
q1=[("DT",{"max_depth":33,"min_samples_split":100}, a2)]

SVM=SVC(kernel='rbf',C=1e3, gamma=0.0001) # SVM
start = time.perf_counter()
SVM.fit(datasets,l)
pred3=SVM.predict(datasets_test)
end = time.perf_counter()
t1.append(end-start)
a3=metrics.accuracy_score(labels_test,pred3)
q1.append(("SVM",{"kernel":"rbf","C":"les3","gamma":0.0001}, a3))

KNN=KNeighborsClassifier(n_neighbors=20,weights='uniform') #KNN
start = time.perf_counter()
KNN.fit(datasets,l)
pred4=KNN.predict(datasets_test)
end = time.perf_counter()
t1.append(end-start)
a4=metrics.accuracy_score(labels_test,pred4)
q1.append(("K-NN",{"n_neighbors":20,"weights":'uniform'}, a4))

MLP=MLPClassifier(alpha=0.05,solver ='adam',batch_size=800,max_iter=200,beta_1=0.85,beta_2=0.7) #MLP
start = time.perf_counter()
MLP.fit(datasets,l)
pred5=MLP.predict(datasets_test)
end = time.perf_counter()
t1.append(end-start)
a5=metrics.accuracy_score(labels_test,pred5)
q1.append(("MLP",{"alpha":0.05, "solver":"adam","batch_size":800,"max_iter":200,"beta_1":0.85,"beta_2":0.7}, a5))

bagging=BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                          n_estimators=9,max_samples=20586,max_features=17) # bagging
start = time.perf_counter()
bagging.fit(datasets,l)
pred6=bagging.predict(datasets_test)
end = time.perf_counter()
t1.append(end-start)
a6=metrics.accuracy_score(labels_test,pred6)
q1.append(("bagging",{"base_estimator":"tree.DecisionTreeClassifier()","n_estimators":9,"max_samples":20586,"max_features":17},a6))

# draw table and show each classifier's parameters, accuracy and training time
table = Texttable()
table.add_rows([["classifier","parameters","accuracy","training time"],
                [q1[0][0],q1[0][1],q1[0][2],t1[0]],
                [q1[1][0],q1[1][1],q1[1][2],t1[1]],
                [q1[2][0],q1[2][1],q1[2][2],t1[2]],
                [q1[3][0],q1[3][1],q1[3][2],t1[3]],
                [q1[4][0],q1[4][1],q1[4][2],t1[4]],             
               ])

print(table.draw())
