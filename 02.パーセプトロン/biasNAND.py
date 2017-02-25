#3
#バイアスを使ったNANDゲートの実装
#ANDの重みとバイアスに、それぞれマイナスをかけるのみ。

import numpy as np

def NAND(x1,x2):
    x=np.array([x1,x2]) #入力信号
    w=np.array([-0.5,-0.5]) #重み
    b= 0.7 #バイアス
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    elif tmp>0:
        return 1

print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))
