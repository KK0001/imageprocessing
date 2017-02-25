#sinとcosのグラフを表示して、それぞれを見やすく編集

import numpy as np
import matplotlib.pyplot as plt

#データの作成
x=np.arange(0,6,0.1)#0から6まで0.1刻みで生成
y1=np.sin(x)
y2=np.cos(x)

#グラフ関係
plt.plot(x,y1,label="sin") #ラベルを"sin"
plt.plot(x,y2,linestyle="--",label="cos") #ラインを破線で描く、ラベルを"cos"
plt.xlabel("x") #x軸のラベル"x"
plt.ylabel("y") #y軸のラベル"y"
plt.title('sin&cos') #グラフのタイトル
plt.legend() #凡例を表示(あったほうがわかりやすい)
plt.show()
