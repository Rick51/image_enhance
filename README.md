# image_enhance

利用pyqt5框架做了个图像增强小工具。包含图像颜色增强，导向滤波，双边滤波等去噪平滑处理，锐化增强，当然还结合了智能检测功能。

python main.py

原图：



![raw_input](https://github.com/Rick51/image_enhance/tree/master/images/10.jpg)



处理过程：



![process](https://github.com/Rick51/image_enhance/tree/master/images/process.jpg)



输出：



![output](https://github.com/Rick51/image_enhance/tree/master/images/final.jpg)



(从其他项目中借用的人脸图片，如有侵犯，还请告知，感谢)

Todo:

1.对于高分辨率图片，某些增强模块比如滤波算法运行缓慢，需要优化；

2.智能检测部分暂时只支持人脸检测，还需继续添加其他模块；

3.工具本来是为了一个调参任务开发，没有特别注意可以合理的关闭整个工具。

