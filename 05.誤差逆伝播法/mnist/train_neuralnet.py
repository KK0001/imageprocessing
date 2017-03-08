#3
#誤差逆伝播法を用いたニューラルネットワークの学習

import numpy as np
from mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#データの読み込み(one-hot表現はTrue)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#2層のニューラルネットワーク
#入力層:784,隠し層:50,出力層:10,がニューロンの数
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#ハイパーパラメータ
iters_num = 10000 #
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = [] #損失
train_acc_list = [] #精度
test_acc_list = [] #

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    #ミニバッチの取得(batch_size=100個ずつデータを無造作に取り出して行う)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配 ここで誤差逆伝播法を用いる。
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #損失関数の値
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1エポックごとに認識精度を表示(1に近づく)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
