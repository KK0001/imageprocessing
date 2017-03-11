#3
#最適化を行うための方法(optimizer)
###
# lr=learning rate
# params=重みパラメータ
# grads=勾配
# v=速度

#SGD
#(確率的勾配降下法)
# W=W-lr*grad
class SGD:
  def __init__(self,lr=0.01):
    self.lr=lr

  def update(self,params,grads):
    for key in params.key():
      params[key]-=self.lr*grads[key]

#Momentum (物理法則の応用を用いた勾配降下法)
# v=v-lr*grads
# W=W+v
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None

    def update(self,params,grads):
        #速度の初期化
        #パラメータと同じ構造のデータをディクショナリ型で保持
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)

        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*gtads[ley]
            params[key]+=self.v[key]

#AdaGrad (パラメータの要素ごとに適応的に更新ステップを調整する方法)
# h=h+grads*grads
# W= W - lr/sqrt(h) * grads
class Adagrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)

        for ley in params.key():
            self.h+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)# 0にならない為の1e-7

#Adam (Momentum + AdaGrad)
#ハイパーパラメータの「バイアス補正」も行われる
