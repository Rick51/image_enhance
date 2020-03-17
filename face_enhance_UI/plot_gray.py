import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_gray(image,image_name):

    image_YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    image_Y, image_U, image_V = cv2.split(image_YUV)
    grayHist = np.zeros((256,),np.uint64)

    flag = "noneed"
    left, right = 0,0

    for i in range(image_Y.shape[0]//6, image_Y.shape[0]*5//6):
        for j in range(image_Y.shape[1]//6, image_Y.shape[1]*5//6):
            if image_Y[i][j] < 80:
                left += 1
    
            elif image_Y[i][j] > 160:
                right +=1

            grayHist[image_Y[i][j]] += 1

    total_pixel_number = image_Y.shape[0] * image_Y.shape[1] * 4 // 9
    if left > (total_pixel_number * 2 // 3) and right < (total_pixel_number // 3):
        flag = "need"

    #print(np.sum(self.grayHist[:50]))
    plt.figure(figsize=(80, 80), frameon=False)
    plt.plot(range(256), grayHist,'r',linewidth=12,c='blue')
    # plt.hist(image_Y.ravel(), bins=256, range=[0, 256])
    # plt.axes().get_yaxis().set_visible(True)
    # plt.axes().get_xaxis().set_visible(False)
    # ax = plt.axes()
    # 隐藏坐标系的外围框线
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    plt.savefig(os.path.join(image_path, '{}_{}.png'.format(image_name.split('.')[0] + '_hist' , flag)), bbox_inches="tight", transparent=True, dpi=100)

image_path = "../image_enhancement/face0"

if __name__ == '__main__':
    
    image_names = [fn for fn in os.listdir(image_path) if fn.endswith('.jpeg')]
    for item in image_names:
        image = cv2.imread(os.path.join(image_path, item))
        plot_gray(image, item)
        print("{} done".format(item))