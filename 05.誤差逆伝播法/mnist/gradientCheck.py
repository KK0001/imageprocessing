#3
#勾配確認

import numpy as np
from mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 入力層:784,隠し層:50,出力層:10,がニューロンの数
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#入力データと教師データ
x_batch = x_train[:3]
t_batch = t_train[:3]

#勾配関係
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
