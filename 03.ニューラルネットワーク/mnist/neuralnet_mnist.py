#3
#MNISTデータセットに対して推論処理を行うニューラルネットワークの実装

import numpy as np
import pickle
from mnist import load_mnist

#ジグモイド関数(すでにNumPy配列に対応)
def sigmoid(x):
    return 1/(1+np.exp(-x))

#ソフトマックス関数
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) #オーバーフロー対策
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

#データセットの取得(画像データを収納)
def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=False)

    return x_test,t_test

#学習済みの重みパラメータの読み込み
def init_network():
    with open("sample_weight.pkl",'rb')as f:
        network=pickle.load(f)

    return network

#ニューラルネットワーク
def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y

#main
x,t=get_data()
network=init_network()

#xを1次元にして、第0層の入力信号として扱う
#xに収納された画像データを1枚ずつfor文で取り出し、
#predict()関数によって分類を行う。
accuracy_cnt=0 #認識精度の分子
for i in range(len(x)):
    y=predict(network,x[i]) #推論し、各要素の確率を出す
    p=np.argmax(y) #最も確率の高い要素のインデックスを取得
    if p==t[i]:
        accuracy_cnt+=1 #答え合わせして、正解だったら認識精度をプラス1

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))
