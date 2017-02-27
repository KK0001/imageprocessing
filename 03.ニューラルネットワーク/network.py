#3
#sigmoid関数によるニューラルネットワークの実装

import numpy as np

#sigmoid関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#最後の出力
def identify_function(x):
    return x

#重みとバイアスの初期化、辞書型で変数networkに収納
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])

    return network

#入力信号が出力信号に変換される関数、出力信号を返す
def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identify_function(a3)

    return y


network=init_network()#変数network
x=np.array([1.0,0.5])#入力信号
y=forward(network,x)#出力信号。forward関数にnetworkとxを引数として出力を取得
print(y)#出力を確認
