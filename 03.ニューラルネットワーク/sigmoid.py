#3
#ジグモイド関数のグラフ化

import numpy as np
import matplotlib.pylab as plt

#ジグモイド関数(すでにNumPy配列に対応)
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)#-5.0から5.0までo.1刻みで生成
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#y軸の範囲を指定
plt.show()
