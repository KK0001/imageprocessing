#3
#ミニバッチ学習
#大量にあるデータから無造作に幾つかのデータを取り出し、学習する

import numpy as np
from mnist import load_mnist

# データの読み込み。one-hot表現にする。
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# データの形状を表示
# この中から、無造作に10枚を取り出したい
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000,10)

# 無造作に10枚を取り出す
###
# np.random.chioce(a,b): 0からa未満までの数の中からランダムにb個の数字を取り出す
#
###
train_size=x_tarin.shape[0] #60000
batch_size=10 #10枚
batch_mask=np.random.choice(train_size,batch_size) #0から60000未満の数から10個取り出す(配列として取り出される)
x_train=x_train[batch_mask]
t_train=t_train[batch_mask]
