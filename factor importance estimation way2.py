# feature importance estimation way2
import numpy as np
import pandas as pa
import time
from texttable import Texttable
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import tree

# data for training
# usecols: columns of features to train the data
# column 0 not chosen for it's gameId,column 1 not chosen for it's creationTime, column 3 not chosen for it's seasonId
# column 4 is the winner
df1=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",
                usecols=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
df2=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/new_data.csv",usecols=[4])
datasets=df1.values
labels=df2.values
l=np.ravel(labels)

# data for testing
df1_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",
                usecols=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
df2_test=pa.read_csv("D:/Hao Ying/大学/2020-2021大二/工业大数据/project_1/test_set_new.csv",usecols=[4])
datasets_test=df1_test.values
labels_test=df2_test.values
l_test=np.ravel(labels_test)
 
DT=tree.DecisionTreeClassifier(max_depth=33,min_samples_split=100)
start = time.perf_counter()
DT.fit(datasets,l)
pred2=DT.predict(datasets_test)
end = time.perf_counter()
t1=[end-start]
a2=metrics.accuracy_score(labels_test,pred2)
q1=[("DT",{"max_depth":"no limitation"}, a2)]

# use feature_importances_ (parameters of DT) to estimate factor importance and visualize
feature_name=['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
              't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
              't2_baronKills','t2_dragonKills','t2_riftHeraldKills','gameduration_new']
feature_importance_df = pa.DataFrame({'name':feature_name,'importance':DT.feature_importances_})
feature_importance_df.sort_values(by = 'importance',ascending=False,inplace = True)
y = feature_importance_df['name']
x = feature_importance_df['importance']
plt.title('Importance of Factors')
plt.xlabel('Importance')
plt.scatter(x,y)
for x1,y1 in zip(x,y):
     plt.text(x1,y1,'%.5f'%x1)
plt.show()
