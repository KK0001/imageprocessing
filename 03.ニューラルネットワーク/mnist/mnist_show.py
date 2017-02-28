#3
#MNIST画像の表示
###
# img = img.reshape(28, 28)
#flatten=True として読み込んだ画像はNumPy配列として1列で収納されている。
#画像の表示には、元の形状である28×28のサイズに再変形(reshape)する必要がある。
###

import numpy as np
from mnist import load_mnist
from PIL import Image

#イメージを表示する関数
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
