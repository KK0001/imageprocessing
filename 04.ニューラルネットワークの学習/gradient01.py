#3
#2次方程式のグラフ化

import numpy as np
import matplotlib.pylab as plt

# y=0.01x^2+0.1x
def function_1(x):
    return 0.01*x**2+0.1*x

x=np.arange(0.0,20.0,0.1)# 0から20まで0.1刻みで生成
y=function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
