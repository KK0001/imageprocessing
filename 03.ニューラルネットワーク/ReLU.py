#3
#ReLU関数のグラフ化

#########
#Memo
# np.maximum(0,x): 0とxを比較して大きい方を出力
# y.astype(np.int): yをnp.int型に変換。
# 真偽値: true:1(非0), false:0
#########

import numpy as np
import matplotlib.pylab as plt

#ジグモイド関数(すでにNumPy配列に対応)
def relu(x):
    return np.maximum(0,x)

x=np.arange(-5.0,5.0,0.1)#-5.0から5.0までo.1刻みで生成
y=relu(x)
plt.plot(x,y)
plt.ylim(-0.1,5.1)#y軸の範囲を指定
plt.show()
