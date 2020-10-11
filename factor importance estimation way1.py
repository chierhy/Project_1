# factors importance estimation way1
import numpy as np
import pandas as pa
import time
from texttable import Texttable

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

q=[] # accuracy
t1=[] # training time
# all factors are columns[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# a[1] means chosen features not include factor 1(game duration) 
a=[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
   [5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21],[5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21],[5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21],
   [5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21],[5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21],[5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21],
   [5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21],[5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21],[5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21],
   [5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21],[5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21],
   [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

for i in range(18):
    # data for training
    # usecols: columns of features to train the data
    # column 0 not chosen for it's gameId,column 1 not chosen for it's creationTime, column 3 not chosen for it's seasonId
    # column 4 is the winner
    df1=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",
                    usecols=a[i])
    df2=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",usecols=[4])
    datasets=df1.values
    prediction=df2.values
    l=np.ravel(prediction)

    # data for testing
    df1_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",
                usecols=a[i])
    df2_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",usecols=[4])
    datasets_test=df1_test.values
    labels_test=df2_test.values
    l_test=np.ravel(labels_test)
       
    KNN=KNeighborsClassifier(n_neighbors=2,weights='uniform') 
    start = time.perf_counter()
    KNN.fit(datasets,l)
    pred3=KNN.predict(datasets_test)
    end = time.perf_counter()
    t1.append(end-start)
    q.append(metrics.accuracy_score(labels_test,pred3))

# q[0] is the accuracy with all classifiers, q[1]~q[17] is the accuracy without one classifier 
table = Texttable()
table.add_rows([["conditons","accuracy","training time"],
                ['all factors',q[0],t1[0]],
                ['no factor 1 (game Duration):',q[17],t1[17]],
                ['no factor 2 (firstBlood):',q[1],t1[1]],
                [ 'no factor 3 (firstTower):',q[2],t1[2]],
                ['no factor 4 (firstInhibitor):',q[3],t1[3]],
                ['no factor 5 (firstBaron):',q[4],t1[4]],
                ['no factor 6 (firstDragon):',q[5],t1[5]],
                ['no factor 7 (firstRiftHerald):',q[6],t1[6]],
                ['no factor 8 (t1_towerKills):',q[7],t1[7]],
                ['no factor 9 (t1_inhibitorKills):',q[8],t1[8]],
                ['no factor 10 (t1_baronKills):',q[9],t1[9]],
                ['no factor 11 (t1_dragonKills):',q[10],t1[10]],
                ['no factor 12 (t1_riftHeraldKills):',q[11],t1[11]],
                ['no factor 13 (t2_towerKills):',q[12],t1[12]],
                ['no factor 14 (t2_inhibitorKills):',q[13],t1[13]],
                ['no factor 15 (t2_baronKills):',q[14],t1[14]],
                ['no factor 16 (t2_dragonKills):',q[15],t1[15]],
                ['no factor 17 (t2_riftHeraldKills):',q[16],t1[16]]               
               ])

print(table.draw())

ss=max(q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17])
mm=min(q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17])
for i in range(17):
    if ss==q[i+1]:
        print('factor',i+1,'has relatively little effect on the result')
    if mm==q[i+1]:
        print('factor',i+1,'has relatively big effect on the result')
