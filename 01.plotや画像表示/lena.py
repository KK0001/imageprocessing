from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

#imreadだと読み込めなかったのでPillowで代用しました。
# img=imread('lena.png')
img = Image.open( 'lena.png' )
plt.imshow(img)

plt.show()
