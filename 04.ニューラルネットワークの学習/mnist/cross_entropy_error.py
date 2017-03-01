#3
#交差エントロピー誤差

import numpy as np

###
# yがニューラルネットワークの出力(softmax)、tが教師データ(one-hot表現)
# 返す値が、誤差となり、値が小さいほど正確さを表す
###
def cross_entropy_error(y,t):
    delta=1e-7 #微小な値
    return -np.sum(t*np.log(y+delta))

#テスト
print("正解は「2」\n出力が「2」のときと、「7」ときの誤差の比較")
#one-hot表現の配列
t=np.array([0,0,1,0,0,0,0,0,0,0]) #正解は「2」
#確率
y1=np.array([0.1 ,0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) #「2」の確率が高い
#誤差
error=cross_entropy_error(y1,t)
print(error)

#確率
y2=np.array([0.1 ,0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]) #「7」の確率が高い
#誤差
error=cross_entropy_error(y2,t)
print(error)
