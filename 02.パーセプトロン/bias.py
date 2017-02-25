#3
#バイアスの使用についての演習

import numpy as np

x=np.array([0,1])#入力信号
w=np.array([0.5,0.5])#重み
b= -0.7#バイアス
print(w*x)#入力と重みの配列の積
print(np.sum(w*x))#w*xの大きさ
print(np.sum(w*x)+b)#バイアスを足した値

#結果はおよそ0.2なので、0以上と分かる。
