#3
#ReLU、Sigmoid、Affine(順伝播で行う内積)、Softmaxのレイヤ
#順伝播と逆伝播をPythonで書く

import numpy as np

#ReLUレイヤ
class ReLU:
    def __init__(self):
        self.mask=None #真偽値をを持つ変数

    #順伝播
    def forward(self,x):
        self.mask=(x<=0) #xが0以下の時True、0より大きい時Falseとなる
        out=x.copy() #浅いコピー
        out[self.mask]=0 #xが0以下のときはout=0とする

        return out

    #逆伝播(0以下のとき、信号は0が掛けられ止まるようになっている)
    def backward(self,dout):
        dout[self.mask]=0 #xが0以下の時はdout=0とする。そうじゃない時は、値をそのまま収納。
        dx=dout

        return dx

#sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.out=None

    #順伝播
    def forward(self,x):
        out =1/(1+exp(-x))
        self.out=out

        return out

    #逆伝播
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out #sigmoidのときの逆伝播の出力

        return dx

#Affineレイヤ(内積関係)
class Affine:
    def __init__(self):
        self.W=W #重み
        self.b=b #バイアス
        self.x=None #入力
        self.dW=None #逆伝播の重み
        self.db=db #逆伝播のバイアス

    #順伝播
    def forward(self,x):
        self.x=x
        out = np.dot(x,self.W)+self.b #入力と重みの内積とバイアスの和

        return out

    #逆伝播
    def backward(self,dout):
        dx=np.dot(dout,self.W.T) #流れてきたdoutと重みWの転置の内積
        self.dW=np.dot(self.x.T,dout) #入力xの転置とdoutの内積
        self.db=np.sum(dout,axis=0) #0番目の軸に対しての総和

        return dx

#Softmaxレイヤ(交差エントロピー誤差も含める)
class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None #損失
        self.y=None #出力
        self.t=None #教師データ

    #順伝播
    def forward(self,x,t):
        self.t=t #教師データ
        self.y=softmax(x) #softman()で正規化
        self.loss=cross_entropy_error(self.y,self.t) #誤差

        return self.loss

    #逆伝播
    def backward(self,dout=1):
        batch_size=self.t.shape[0] #教師データの個数
        dx=(self.y-self.t)/batch_size #差を個数で割ったときの正確さを表す

        return dx
