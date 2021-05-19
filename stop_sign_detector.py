import matplotlib
import os
import cv2 as cv2
import numpy as np
import math
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import pickle
    
class StopSignDetector():
    
    #def __init__(self):
        #read saved iid data
        #raise NotImplementedError
    def segment_image(self, img):
        def sigmoid(z):
            return 1 / (1 + math.exp(-z))
        a = pickle.load(open("/Users/clown/PycharmProjects/pythonProject/a.p", "rb" ))   # X: rgb data for training
        b = pickle.load(open("/Users/clown/PycharmProjects/pythonProject/b.p", "rb" ))   # Y: corresponding labels
        a = np.reshape(a,(30,3)) #15 images for training, each image pick 1 stop sign red and the other similiar but not red color.
        b = np.reshape(b,(30,1)) # corresponding 1 and -1 labels
        w = np.zeros((3,1)) # initialize omega  4x1
        alpha = 0.001   # initialize learning rate constant
        u = np.zeros((3,1))  #w=w+au
        ############logistic regression###########  MLE
        for iteration in range(0,10):
            for i in range(0,30):
                k = a[i]
                newx = np.reshape(k,(3,1))
                xT = np.transpose(newx)
                xTw = int(np.dot(xT,w))
                c = int(b[i])*xTw
                #print(c)
                u = u + np.dot(int(b[i]),newx) * (1-sigmoid(c))
            w = w + alpha * u
         
         #####Testing#######
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ###downsize image
        size = len(img) * len(img[0])
        ###downsize image when the size of image is too big
        if size > 1000 * 1000:
            scale_percent = 40 # 40 percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        width = len(img)
        high = len(img[0])
        boundary = np.zeros((width,high))
        mask_img = np.zeros((width,high))

        for i in range(0,width):
            for j in range(0,high):
                l=img[i,j,:]
                new_test=np.reshape(l,(3,1))
                new_testT=np.transpose(new_test)
                boundary[i,j] = np.dot(new_testT,w)
                if boundary[i,j] < 0:  #use boudary to get binary mask image
                    mask_img[i,j] = 0
                else:
                    mask_img[i,j] = 1
        #raise NotImplementedError
        return mask_img
        
    def get_bounding_box(self, img):
        ####get bounding box#####
        mask_img = self.segment_image(img)
        mask_img = np.uint8(mask_img)
        contours, hier = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(mask_img.shape)
        contour = cv2.drawContours(image, contours, -1, (255,0,0), 5)
        contour = np.uint8(contour)
        contour = label(contour)
        row,col = image.shape
        boxes=[]
        for region in regionprops(contour):
            if region.area > 300: #skip small red area
                if region.eccentricity < 0.7: #skip non-circle like red area
                    min_row, min_col, max_row, max_col = region.bbox
                    boxes.append([min_col, row - max_row, max_col, row - min_row])
                    print(min_row, min_col, max_row, max_col)  #corrdinate
                    img = cv2.rectangle(img,(min_col, min_row),(max_col, max_row),(0,255,0),3)
        #raise NotImplementedError
        return boxes

if __name__ == '__main__':
    folder = "/Users/clown/PycharmProjects/pythonProject/trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)
        #plt.imshow(mask_img)
        #(2) Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)
        #plt.imshow(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

