#3
#交差エントロピー誤差を、バッチ対応する

#yはneural_networkの出力、tは教師データ
#yの次元数が1なら、データの形状を整形する。
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size) #1行(t.size)列に変換
        y=y.reshape(1,y.size) #1行(y.size)列に変換
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y))/batch_size
