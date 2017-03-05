#3
#順伝播と逆伝播をPythonで(乗算ノード)

class AddLayer:
    def __init__(self):
        pass#何も行わない
        #共通して使う変数的なのが無いので

    #順伝播(z=x+y)
    def forward(self,x,y):
        out=x+y
        return  out

    #逆伝播
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
