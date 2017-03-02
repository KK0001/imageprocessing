#3
#偏微分

import numpy as np

#2変数関数 y=x0**2+x1**2
def function_2(x):
    return x[0]**2+x[1]**2
    # または return np.sum(x**2)

#引数のfは関数,xはNumPy配列を想定
def numerical_gradient(f,x):
    h=1e-4 # 0.0001
    grad=np.zeros_like(x) #xと同じ形状の配列を生成(要素はすべて0で生成)

    for idx in range(x.size):
        tmp_val=x[idx]
        # f(x+h)の計算
        x[idx]=tmp_val+h
        fxh1=f(x)

        # f(x-h)の計算
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val # 値を元に戻す

    return grad

#テスト
#2変数関数 y=x0**2+x1**2 のときの、勾配
#点(3, 4)
print(numerical_gradient(function_2,np.array([3.0,4.0])))
#点(0,2)
print(numerical_gradient(function_2,np.array([0.0,2.0])))
#点(3,0)
print(numerical_gradient(function_2,np.array([3.0,0.0])))
