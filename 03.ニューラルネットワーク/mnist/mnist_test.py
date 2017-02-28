#3
#MNISTのテスト

from mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)

#データの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
