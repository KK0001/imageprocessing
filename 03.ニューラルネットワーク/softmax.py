#3
#ソフトマックス関数

###
# def softmax(a):
#     exp_a=np.exp(a)
#     sum_exp_a=np.sum(exp_a)
#     y=exp_a/sum_exp_a
#
#     return y
#以上でもソフトマックス関数の再現はできるが、a=1000などの場合にexpが無限大となり
#オーバーフローしてしまうので、対策が必要である。
###

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) #オーバーフロー対策
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y
