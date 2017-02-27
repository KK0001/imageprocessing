#3
#ステップ関数のグラフ化

#########
#Memo
# astype(np.int): 引数に希望する型にnp.intを指定。
# y.astype(np.int): yをnp.int型に変換。
# 真偽値: true:1(非0), false:0
#########

import numpy as np
import matplotlib.pylab as plt

#xが0より大きかったら1を、0以下だったら0を返す関数(NumPy配列に対応)
def step_function(x):
    y=x>0#配列xの中身がx>0かを判別、真偽値がyに収納される。
    return y.astype(np.int)#yの真偽値を0,1に変換してその値をreturnする。

x=np.arange(-5.0,5.0,0.1)#-5.0から5.0までo.1刻みで生成
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#y軸の範囲を指定
plt.show()
