#3
#勾配法をPythonで実装
#関数の勾配をnumerical_gradientで求め、その勾配に学習率を掛けた値で更新する処理をstep_numの数繰り返す。
#gradient_descent()を使えば、関数の極小値を求めることができる。あるいは場合によると、最小値を求めることができる。

import numpy as np

# 勾配を返す関数
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

# 勾配降下法
# f:最適化したい関数、init_x:初期値、lr:学習率(learning rate)、step_num:勾配法による繰り返しの数
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad

    return x

#テスト: f(x0,x1)=0**2+x1**2 の最小値を勾配法で求める。
#2変数関数 y=x0**2+x1**2
def function_2(x):
    return x[0]**2+x[1]**2
    # または return np.sum(x**2)

init_x=np.array([-3.0,4.0])
ans=gradient_descent(function_2,init_x,lr=0.1,step_num=100)
print(ans)
