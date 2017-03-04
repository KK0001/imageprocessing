#3
#簡単なニューラルネットワークを例にして、勾配を求める

import numpy as np

#softmax()関数
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) #オーバーフロー対策
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

#交差エントロピー誤差
def cross_entropy_error(y,t):
    delta=1e-7 #微小な値
    return -np.sum(t*np.log(y+delta))

# 勾配を返す関数(配布されているもの、改良したもの)
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

#簡単なニューラルネットワーク
#形状が2×3の重みパラメータを1つだけインスタント変数として持つ
class simpleNet:
    def __init__(self):
        #重み
        self.W=np.random.randn(2,3)#ガウス分布で初期化

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)

        return loss

#テスト
net=simpleNet() #simpleNetを読み込む
print(net.W) #重みを表示
x=np.array([0.6,0.9]) #配列の生成
p=net.predict(x) #predict()関数で積を計算
print(p)
print(np.argmax(p)) #pの最大値のインデックス
t=np.array([0,0,1]) #正解ラベルの用意
print(net.loss(x,t)) #損失関数loss()

#勾配を求める
f=lambda w:net.loss(x,t)
dW=numerical_gradient(f,net.W)
print(dW)
