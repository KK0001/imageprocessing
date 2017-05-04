#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 参考: http://qiita.com/KojiOhki/items/ff6ae04d6cf02f1b6edf

## MNISTデータの読み込み
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

## 回帰の実装(入力、重み、バイアス、出力などを実装)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b) # matmulでxとWを掛けた後にbを加え、softmaxを適用。

## 訓練(モデルの定義: 良いモデルあるいは悪いモデルとは何かを定義する。)
# 正解を入力する新しいプレースホルダー
y_ = tf.placeholder(tf.float32, [None, 10])
# 交差エントロピー
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 勾配降下アルゴリズムを使用して学習率 0.5 で cross_entropy を最小化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初期化(TensorFlowでのおまじない)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 訓練(1000回)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 予測がどれだけ一致するかをチェック
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# フロート型にキャストして、平均値を取る
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 精度を表示
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
