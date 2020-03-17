import os,sys

import PyQt5
from PyQt5 import Qt
from PyQt5 import QtCore,QtWidgets,QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QGraphicsRectItem, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize, Qt

import cv2
import numpy as np
from matplotlib import pyplot as plt
from eef import process
import window_ui


class MainWindow():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.raw_image = None
        self.ui = window_ui.UI_MainWindow()
        self.ui.setup_ui(MainWindow)
        self.action_connect()
        MainWindow.show()
        sys.exit(app.exec_())

# 信号槽绑定
    def action_connect(self):
        self.ui.action.triggered.connect(self.open_file)
        self.ui.action_18.triggered.connect(self.parse_multi_files)
        self.ui.action_2.triggered.connect(self.save_file)
        self.ui.action_5.triggered.connect(self.recover_img)
        self.ui.action_8.triggered.connect(self.beauty_img_auto)

        #直方图均衡化
        self.ui.horizontalSlider_10.sliderReleased.connect(self.slider_change) #
        self.ui.horizontalSlider_10.sliderReleased.connect(self.show_histogram)

        #限制对比度自适应直方图均衡化
        self.ui.horizontalSlider_9.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_9.sliderReleased.connect(self.show_histogram)
        
        #多重曝光融合
        self.ui.horizontalSlider_7.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_7.sliderReleased.connect(self.show_histogram)

        #导向滤波器磨皮
        self.ui.horizontalSlider_4.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_4.sliderReleased.connect(self.show_histogram)

        #双边滤波器
        self.ui.horizontalSlider.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider.sliderReleased.connect(self.show_histogram)

        #USM锐化
        self.ui.horizontalSlider_8.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_8.sliderReleased.connect(self.show_histogram)

        #亮度调节（线性）
        self.ui.horizontalSlider_12.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_12.sliderReleased.connect(self.show_histogram)

        #对比度调节（线性）
        self.ui.horizontalSlider_2.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_2.sliderReleased.connect(self.show_histogram)

        #饱和度调节
        self.ui.horizontalSlider_6.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_6.sliderReleased.connect(self.show_histogram)

        #Laplace锐化
        self.ui.horizontalSlider_13.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_13.sliderReleased.connect(self.show_histogram)

        #Sobel锐化
        #self.ui.horizontalSlider_11.sliderReleased.connect(self.slider_change)

        #人脸检测
        self.ui.checkbox_21.stateChanged.connect(self.face_detect)
        #人形检测
        #self.ui.checkbox_22.stateChanged.connect(self.person_detect)
        #关键点检测
        #self.ui.checkbox_23.stateChanged.connect(self.landmark_detect)


    def initial_value(self):
        self.calculated = False

        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider_2.setValue(10)
        # self.ui.horizontalSlider_3.setValue(0)
        self.ui.horizontalSlider_4.setValue(0)
        # self.ui.horizontalSlider_5.setValue(0)
        self.ui.horizontalSlider_6.setValue(0)
        self.ui.horizontalSlider_7.setValue(0)
        self.ui.horizontalSlider_8.setValue(0)
        self.ui.horizontalSlider_9.setValue(0)
        self.ui.horizontalSlider_10.setValue(0)
        #self.ui.horizontalSlider_11.setValue(0)
        self.ui.horizontalSlider_12.setValue(0)
        self.ui.horizontalSlider_13.setValue(0)
        #self.ui.horizontalSlider_14.setValue(0)

        # self.horizontalSlider    = self.ui.horizontalSlider.value()
        # self.horizontalSlider_2  = self.ui.horizontalSlider_2.value()
        # self.horizontalSlider_4  = self.ui.horizontalSlider_4.value()
        # self.horizontalSlider_6  = self.ui.horizontalSlider_6.value()
        # self.horizontalSlider_7  = self.ui.horizontalSlider_7.value()
        # self.horizontalSlider_8  = self.ui.horizontalSlider_8.value()
        # self.horizontalSlider_9  = self.ui.horizontalSlider_9.value()
        # self.horizontalSlider_10 = self.ui.horizontalSlider_10.value()
        # self.horizontalSlider_12 = self.ui.horizontalSlider_12.value()

    def initial_value_auto(self):
        self.calculate_grayhist()
        
        if np.sum(self.grayHist[-150:]) < 25:
            self.ui.horizontalSlider_2.setValue(16)
        elif np.sum(self.grayHist[-100:]) < 25:
            self.ui.horizontalSlider_2.setValue(14)
        elif np.sum(self.grayHist[-50:]) < 25:
            self.ui.horizontalSlider_2.setValue(13)
        else:
            self.ui.horizontalSlider_2.setValue(12)
            
        self.ui.horizontalSlider_12.setValue(12)
        self.ui.horizontalSlider_6.setValue(10)
        self.ui.horizontalSlider_4.setValue(30)
        self.ui.horizontalSlider_8.setValue(10)

    def open_file(self):
        image_name = QFileDialog.getOpenFileName(None, '打开文件', './', ("Images (*.png *.bmp *.jpg *.jpeg)"))
        if image_name[0]:
            image = cv2.imdecode(np.fromfile(image_name[0],dtype=np.uint8),-1)
            image_YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV) 
            self.raw_image = image.copy()
            self.last_image = image.copy()
            self.current_image = image.copy()
            self.skinimage = np.zeros(self.raw_image.shape)
            self.image_Y,self.image_U,self.image_V = cv2.split(image_YUV)
            self.show_image()
            self.show_histogram()
        self.initial_value()

    def save_file(self):
        image_name = QFileDialog.getSaveFileName(None, '打开文件', './', ("Images (*.png *.bmp *.jpg *.jpeg)"))
        if image_name[0]:
            cv2.imwrite(image_name[0], self.current_image)

    def recover_img(self):
        if self.raw_image is None:
            return 0
        self.current_image = self.raw_image
        self.show_image()
        self.show_histogram()
        self.initial_value()

    def beauty_img_auto(self):
        if self.raw_image is None:
            return 0
        self.initial_value_auto()
        self.slider_change()
        self.show_histogram()

    def parse_multi_files(self):
        image_dir = QFileDialog.getExistingDirectory(None,"选取文件夹", "./")
        if image_dir:
            save_dir = os.path.join(image_dir,"../{}_results".format(image_dir.split('/')[-1]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_name_list = [fn for fn in os.listdir(image_dir)]
            for image_name in image_name_list:
                image = cv2.imdecode(np.fromfile(os.path.join(image_dir,image_name),dtype=np.uint8),-1)
                if image is None:
                    continue
                self.raw_image = image.copy()
                self.current_image = image.copy()
                self.initial_value_auto()
                self.slider_change()
                final_image = self.current_image
                cv2.imwrite(os.path.join(save_dir,image_name.split('.')[0] + '_enhance.jpg'),final_image)

    def show_image(self):
        image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        width,height,_ = image.shape
        wh_ratio = width / height
        scene_ratio = self.ui.graphicsView.width() / self.ui.graphicsView.height()
        if wh_ratio > scene_ratio:
            width_new = int(self.ui.graphicsView.width())
            height_new = int(self.ui.graphicsView.width() / wh_ratio)
        else:
            width_new = int(self.ui.graphicsView.height() * wh_ratio)
            height_new = int(self.ui.graphicsView.height())
        image_resize = cv2.resize(image,(width_new-5, height_new-5))
        H,W,C = image_resize.shape
        bytesPerline = W * 3
        q_image = QImage(image_resize.data, W, H, bytesPerline, QImage.Format_RGB888)
        self.scene = QGraphicsScene()
        pix = QPixmap(q_image)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.setScene(self.scene)

    def calculate_grayhist(self):
        image = self.current_image
        image_YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        self.image_Y, self.image_U, self.image_V = cv2.split(image_YUV)
        self.grayHist = np.zeros((256,),np.uint64)
        for i in range(self.image_Y.shape[0]):
            for j in range(self.image_Y.shape[1]):
                self.grayHist[self.image_Y[i][j]] += 1


    def show_histogram(self):
        if self.raw_image is None:
            return 0
        image = self.current_image
        self.calculate_grayhist()
        #print(np.sum(self.grayHist[:50]))
        plt.figure(figsize=((self.ui.tab_3.width()-10)/100, (self.ui.tab_3.width()-60)/100), frameon=False)
        #plt.plot(range(256),self.grayHist,'r',linewidth=2,c='blue')
        plt.hist(self.image_Y.ravel(), bins=256, range=[0, 256])
        plt.axes().get_yaxis().set_visible(False)
        #plt.axes().get_xaxis().set_visible(False)
        ax = plt.axes()
        # 隐藏坐标系的外围框线
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig('Hist.png', bbox_inches="tight", transparent=True, dpi=100)
        pix = QPixmap("Hist.png")
        self.ui.label.setPixmap(pix)
        self.ui.label_2.setPixmap(pix)
        self.ui.label_3.setPixmap(pix)

    def detect_skin(self):
        image = self.raw_image
        h, cols, channals = image.shape
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                B = int(image[i,j,0])
                G = int(image[i,j,1])
                R = int(image[i,j,2])
                if (abs(R - G) > 15) and (R > G) and (R > B):
                    if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                        self.skinimage[i,j,:] = 1
                    elif (R > 220) and (G > 210) and (B > 170):
                        self.skinimage[i,j,:] = 1

    def hist_equalize(self):
        image = self.current_image#.copy()
        hist_for_Y = 0
        hist_for_bgr = 1
        if hist_for_Y:
            image_YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
            self.image_Y,self.image_U,self.image_V = cv2.split(image_YUV)
            # h,w = self.image_Y.shape
            # #计算累加直方图
            # sumHist = np.zeros((256,),np.uint32)
            # sumHist[0] = self.grayHist[0]
            # for pixel_value in range(1,256):
            #     sumHist[pixel_value] = sumHist[pixel_value - 1] + self.grayHist[pixel_value]
            # #计算输入灰度级 和 输出灰度级的映射关系
            # cofficient = 256.0/(h*w)
            # output = np.zeros((256,),np.uint8)
            # for p in range(256):
            #     output[p] = max(0,(cofficient * sumHist[p] - 1))
            # equalizeHistImage = np.zeros(self.image_Y.shape,np.uint8)
            # for i in range(h):
            #     for j in range(w):
            #         equalizeHistImage[i,j] = output[self.image_Y[i,j]]
            # self.image_Y = equalizeHistImage
            self.image_Y = cv2.equalizeHist(self.image_Y)
            image_YUV_new = cv2.merge([self.image_Y,self.image_U,self.image_V])
            image = cv2.cvtColor(image_YUV_new,cv2.COLOR_YUV2BGR)
        elif hist_for_bgr:
            for i in range(3):
                image[:,:,i] = cv2.equalizeHist(image[:,:,i])
        self.current_image = image


    #限制对比度自适应直方图均衡化
    def clahe(self):
        image = self.current_image#.copy()
        hist_for_Y = 0
        hist_for_bgr = 1
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (self.ui.horizontalSlider_9.value(),self.ui.horizontalSlider_9.value()))
        if hist_for_Y:
            image_YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
            self.image_Y,self.image_U,self.image_V = cv2.split(image_YUV)
            self.image_Y = clahe.apply(self.image_Y)
            image_YUV_new = cv2.merge([self.image_Y,self.image_U,self.image_V])
            self.current_image = cv2.cvtColor(image_YUV_new,cv2.COLOR_YUV2BGR)
        elif hist_for_bgr:
            for i in range(3):
                image[:,:,i] = clahe.apply(image[:,:,i])
            self.current_image = image

    def eef_process(self):
        image = self.current_image#.copy()
        image = process(image)
        self.current_image = image

    def adjust_brightness(self):
        image = self.current_image#.copy()
        alpha = self.ui.horizontalSlider_2.value() / 10
        beta = self.ui.horizontalSlider_12.value()
        image = alpha * image + beta
        image[image > 255] = 255
        image[image < 0] = 0
        self.current_image = image.astype(np.uint8)

    def adjust_saturate(self):
        image = self.current_image#.copy()
        increment = self.ui.horizontalSlider_6.value() / 100
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                max_pixel_value = int(max(image[i,j]))
                min_pixel_value = int(min(image[i,j]))
                delta = (max_pixel_value - min_pixel_value) / 255.0
                value = (max_pixel_value + min_pixel_value) / 255.0
                if delta == 0:
                    continue
                L = value / 2.0
                if L < 0.5:
                    s = delta/(value)
                else:
                    s = delta/(2 - value)
                if increment > 0:
                    if increment + s >= 1:
                        alpha = s
                    else:
                        alpha = 1 - increment
                    alpha = 1/alpha - 1
                else:
                    alpha = increment
                image[i,j,:] = image[i,j,:] + (image[i,j,:] - L * 255.0) * alpha
        self.current_image = image

    def guidefilter(self):
        scale = 1
        eps = self.ui.horizontalSlider_4.value()*0.0001
        side_size = 2 #self.ui.horizontalSlider_4.value() #7
        image = self.current_image#.copy()
        h,w,_ = image.shape
        image = cv2.resize(image,(int(w/scale),int(h/scale)))
        
        I,P = image.copy(),image.copy()
        T,P = I.astype(np.float64),P.astype(np.float64)
        winSize = ( max(1,int((2*side_size+1)/scale)) , max(1,int((2*side_size+1)/scale)+1)) #(7,7)
        I = I / 255.0
        P = P / 255.0
        mean_I = cv2.blur(I,winSize)
        mean_P = cv2.blur(P,winSize)

        mean_II = cv2.blur(I*I, winSize)
        mean_IP = cv2.blur(I*P, winSize)

        var_I = mean_II - mean_I * mean_I
        cov_IP = mean_IP - mean_I * mean_P

        a = cov_IP / (var_I + eps)
        b = mean_P - a * mean_I
        mean_a = cv2.blur(a,winSize)
        mean_b = cv2.blur(b,winSize)
        q = mean_a * I + mean_b
        result = np.zeros(I.shape,np.uint8)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                for c in range(I.shape[2]):
                    result[i,j,c] = min(q[i,j,c]*255,255)
        
        result = cv2.resize(result,(w,h))
        self.current_image = result

        # winSize = (2*side_size+1,2*side_size+1)
        
        # mean_I = cv2.boxFilter(I, ksize = winSize, ddepth=-1, normalize=True)
        # mean_P = cv2.boxFilter(P, ksize = winSize, ddepth=-1, normalize=True)
        # corr_I = cv2.boxFilter(I*I,ksize = winSize,ddepth=-1,normalize=True)
        # corr_IP = cv2.boxFilter(I*P,ksize = winSize,ddepth=-1,normalize=True)

        # var_I = corr_I- mean_I*mean_I
        # cov_IP = corr_IP - mean_I*mean_P

        # a = cov_IP / (var_I + eps)
        # b = mean_P - a*mean_I

        # mean_a = cv2.boxFilter(a, ksize = winSize, ddepth=-1,normalize=True)
        # mean_b = cv2.boxFilter(b,ksize = winSize, ddepth=-1, normalize=True)

        # q = mean_a * I + mean_b
        # q = q.astype(np.uint8)
        # self.current_image = q

    #双边滤波, 太耗时间，先不予考虑。
    def bilfilter(self):
        image = self.current_image#.copy()
        value2 = 11 - self.ui.horizontalSlider.value()
        value1 = self.ui.horizontalSlider.value()
        dx = value1 * 5
        fc = value1 * 12.5
        p = 0.60
        temp1 = cv2.bilateralFilter(image, dx, fc, fc)
        self.current_image = temp1


    def usm(self):
        image = self.current_image#.copy()
        amount = self.ui.horizontalSlider_8.value()
        expand_image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_REPLICATE)
        height,width = expand_image.shape[:2]
        for c in range(3):
            for x in range(1,height-1):
                for y in range(1,width-1):
                    HighPass = 4*(expand_image[x,y,c]) - expand_image[x-1,y,c] - expand_image[x+1,y,c] \
                                                          - expand_image[x,y-1,c] - expand_image[x,y+1,c]
                    value = image[x-1,y-1,c] + amount * HighPass//100
                    if value > 255:
                        value = 255
                    elif value < 0:
                        value = 0
                    image[x-1,y-1,c] = value
        self.current_image = image

    def laplace(self):
        image = self.current_image
        #opencv
        if self.ui.horizontalSlider_13.value() == 1:
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        elif self.ui.horizontalSlider_13.value() == 2:
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        image = cv2.filter2D(image,-1,kernel)
        #手动实现
        # height,width,_ = image.shape
        # image = image.astype(np.float64)
        # new_image = np.zeros(image.shape)
        # for i in range(2,height-1):
        #     for j in range(2,width-1):
        #         #负数在uint8格式会变成正数
        #         pixel = image[i+1,j,:] + image[i-1,j,:] + image[i,j-1,:] + image[i,j+1,:] - 4*image[i,j,:]
        #         new_image[i,j,:] = pixel
        # image =  image - new_image
        # image[image < 0] = 0
        # image[image > 255] = 255 
        # image = image.astype(np.uint8)
        self.current_image = image

    def sobel(self):
        image = self.current_image
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #手动实现
        # sobel_x = [1,0,-1,2,0,-2,1,0,-1]
        # sobel_y = [-1,-2,-1,0,0,0,1,2,1] 
        # h,w,_ = image.shape
        # image_sobel = np.zeros((h,w))
        # temp_x = [0]*9
        # temp_y = [0]*9
        # for i in range(1,h-1):
        #     for j in range(1,w-1):
        #         for k in range(3):
        #             for l in range(3):
        #                 temp_x[k*3+l] = image[i-1+k,j-1+l]*sobel_x[k*3+l]
        #                 temp_y[k*3+l] = image[i-1+k,j-1+l]*sobel_y[k*3+l]
        # to be continue
        
        #opencv
        image = image.astype(np.float64)
        #for c in range(3):
        x = cv2.Sobel(image_gray,cv2.CV_16S,1,0)
        y = cv2.Sobel(image_gray,cv2.CV_16S,0,1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        
        #image_gray = cv2.add(image_gray,Sobel)
        Sobel = cv2.cvtColor(Sobel,cv2.COLOR_GRAY2BGR)        
        
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype(np.uint8)
        self.current_image = image

    def face_detect(self,state):
        if self.raw_image is None:
            return 0
        if state == Qt.Checked:
            image = self.current_image.copy()
            self.last_image = self.current_image.copy()
            face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,h,w) in faces:
                image = cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255),1)
            self.current_image = image
            self.show_image()
        else:
            self.current_image = self.last_image
            self.show_image()

    def person_detect():
        pass

    def landmark_detect():
        pass

    def slider_change(self):
        if self.raw_image is None:
            return 0
        self.current_image = self.raw_image.copy()
        #if not self.calculated: 
        #    self.detect_skin()
        #    self.calculated = True

        #直方图均衡化
        if  self.ui.horizontalSlider_10.value():
            self.hist_equalize()

        #限制对比度自适应直方图均衡化
        if self.ui.horizontalSlider_9.value():
            self.clahe()
    
        #EEF
        if self.ui.horizontalSlider_7.value():
            self.eef_process()

        #亮度
        if self.ui.horizontalSlider_12.value():
            self.adjust_brightness()

        #对比度
        if self.ui.horizontalSlider_2.value() > 10:
            self.adjust_brightness()

        #饱和度
        if self.ui.horizontalSlider_6.value():
            self.adjust_saturate()

        #导向滤波器磨皮
        if self.ui.horizontalSlider_4.value():
            self.guidefilter()

        #双边滤波器
        if self.ui.horizontalSlider.value():
            self.bilfilter()

        #USM锐化
        if self.ui.horizontalSlider_8.value():
            self.usm()

        #Laplace锐化
        if self.ui.horizontalSlider_13.value():
            self.laplace()

        #Sobel锐化
        #if self.ui.horizontalSlider_11.value():
        #    self.sobel()

        self.show_image()

if __name__ == "__main__":
    MainWindow()