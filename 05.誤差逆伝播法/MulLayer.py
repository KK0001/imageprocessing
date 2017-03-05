#3
#順伝播と逆伝播をPythonで(乗算ノード)

#乗算ノード
class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None

    #順伝播(z=x*y)
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y

        return out

    #逆電波(z=x*yより)
    def backward(self,dout):
        #doutは、上流から来る値
        dx=dout*self.y
        dy=dout*self.x

        return dx,dy

##テスト
# #100円のリンゴを2個買うとき(消費税10%とする)
# apple=100
# apple_num=2
# tax=1.1
#
# #layer(リンゴ用と消費税用)
# mul_apple_layer=MulLayer()
# mul_tax_layer=MulLayer()
#
# #forward
# apple_price=mul_apple_layer.forward(apple,apple_num)
# price=mul_tax_layer.forward(apple_price,tax)
#
# print(price) #220
#
# #backward
# dprice=1
# dapple_price,dtax=mul_tax_layer.backward(dprice)
# dapple,dapple_num=mul_apple_layer.backward(dapple_price)
#
# print(dapple,dapple_num,dtax) #2.2 110 200
