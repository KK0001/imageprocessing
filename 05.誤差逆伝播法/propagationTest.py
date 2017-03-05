#3
#伝播法をPythonで実装する

from MulLayer import *
from AddLayer import *

#100円のリンゴを2個、150円のみかんを3個買うとき(消費税10%とする)の伝播

#変数
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1

#layer(リンゴ用と消費税用)
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange=AddLayer()
mul_tax_layer=MulLayer()

#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange,orange_num)
all_price=add_apple_orange.forward(apple_price,orange_price)
price=mul_tax_layer.forward(all_price,tax)

#backward
dprice=1
dall_price,dtax=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange.backward(dall_price)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)

print("price",int(price)) #715

print("dapple,dapple_num")
print(dapple,int(dapple_num)) #2.2 110 200
print("dorange,dorange_num")
print(dorange,int(dorange_num)) #2.2 110 200
print("tax",tax)
