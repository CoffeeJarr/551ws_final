import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import os, cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import glob

a=[]    # sum matrix of all x
b=[]    # sum matrix of all y
img_dir = "mytrainset/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*.jpg')
files = glob.glob(data_path)
for f1 in files:
    x=np.zeros((2,3)) #2x3  each image choose two rgb
    y=np.zeros(2)   #n
    img_bgr = cv2.imread(f1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_rgb)
    img_roi1 = RoiPoly(color='r')
    plt.imshow(img_rgb)
    img_roi2 = RoiPoly(color='g')
    img_mask1 = img_roi1.get_mask(img_gray)
    img_mask2 = img_roi2.get_mask(img_gray)
    for i in range(0,len(img_mask1)):
        for j in range(0,len(img_mask1[0])):
            if img_mask1[i,j] == True:
                x[0] = img_rgb[i,j,:]
                y[0] = 1 #red
            if img_mask2[i,j] == True:
                x[1] = img_rgb[i,j,:]
                y[1] = -1 #not red
    a.append(x)
    b.append(y)

a=np.array(a)
b=np.array(b)
pickle.dump( a, open( "a.p", "wb" ) )
pickle.dump( b, open( "b.p", "wb" ) )
