import cv2
import numpy as np
import matplotlib.pyplot as plt



def colorRead(path):
    return cv2.imread(path,cv2.IMREAD_COLOR)#默认读取bgr格式

def grayRead(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def showInfo(image):
    print("[opencv] image.shape" + str(image.shape))
    print("[opencv] image.size" + str(image.size))#像素个数
    print("[opencv] image.dtype" + str(image.dtype))

def saveImg(imgPath,img):
    cv2.imwrite(imgPath,img)

def cutpic(img):
    cat = img[0:50,0:200]
    return cat

def cutChannel(img):
    b,g,r=cv2.split(img)
    return b,g,r

def showImg(time,img,imgName):
    cv2.imshow(imgName,img)
    cv2.waitKey(time)#任意键终止
    cv2.destroyAllWindows()


def padding(img):
    top,bottom,left,right = (50,50,50,50)
    re = cv2.copyMakeBorder(img,top,bottom,left,right,borderTypee = cv2.BORDER_REPLICATE)
    #填充类型:REFLECT REFLECT_101 WRAP CONSTENT
    return re
def thre(img):
    ret,dst = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #ret,dst = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#反转
    #ret,dst = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)#大于阈值的部分设置为阈值，否则不变
    #ret,dst = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)#大于阈值的部分不变，否则设置为0
    #ret,dst = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    return dst

def smooth(img):
    #均值滤波
    blur = cv2.blur(img,(3,3))#求均值，赋值到最中间的像素
    #方框滤波
    box = cv2.boxFilter(img,-1,(3,3),normalize=True)#-1表示颜色通道不变 normalize是否归一化，true值时与均值滤波相同
    #高斯滤波
    Gaussian = cv2.GaussianBlur(img,(5,5),1)#越近的像素权重越大
    #中值滤波
    median = cv2.medianBlur(img,5)#取区域内的中值
    #return np.hstack((blur,box,Gaussian,median))
    #return blur
    #return box
    #return Gaussian
    return median

def corrosionOperation(img):#腐蚀操作
    kernel = np.ones((5,5),np.uint8)#返回一个值全是1的数组，zero函数同理
    erosion = cv2.erode(img,kernel,iterations = 1)
    return erosion

def expandOperation(img):#膨胀运算
    kernel = np.ones((3,3),np.uint8)
    dige=cv2.dilate(img,kernel,iterations = 1)
    return dige

def openCloseOperation(img):
    kernel = np.ones((5,5),np.uint8)
    #open 先腐蚀后膨胀
    #close 前膨胀后腐蚀
    opOrCl = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#CLOSE
    return opOrCl

def gradientOperation(img):
    #梯度运算 膨胀-腐蚀 得到边界
    kernel = np.ones((7,7),np.uint8)
    #dilate = cv2.dilate(img,kernel,iterations = 5)
    #erosion = cv2.erode(img,kernel,iterations = 5)
    #res = np.hstack((dilate,erosion))
    return cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

def topHat(img):
    #原始-开运算（剩下毛刺）
    kernel = np.ones((7,7),np.uint8)
    return cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

def blackHat(img):
    #闭运算-原始输入（轮廓）
    kernel = np.ones((7,7),np.uint8)
    return cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
#梯度检测

def sobel(img):
    dstx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 3)#cv2,CV_64F可以表示负数 1，0选择x或者y
    dsty = cv2.Sobel(img,cv2.CV_64F,0,1,ksize = 3)#cv2,CV_64F可以表示负数 1，0选择x或者y
    dstx = cv2.convertScaleAbs(dstx)#取绝对值
    dsty = cv2.convertScaleAbs(dsty)
    return cv2.addWeighted(dstx,1,dsty,1,0)

def scharr(img):
    dstx = cv2.Scharr(img,cv2.CV_64F,1,0)#cv2,CV_64F可以表示负数 1，0选择x或者y
    dsty = cv2.Scharr(img,cv2.CV_64F,0,1)#cv2,CV_64F可以表示负数 1，0选择x或者y
    dstx = cv2.convertScaleAbs(dstx)#取绝对值
    dsty = cv2.convertScaleAbs(dsty)
    return cv2.addWeighted(dstx,1,dsty,1,0)

def laplacian(img):
    lap = cv2.Laplacian(img,cv2.CV_64F)
    return cv2.convertScaleAbs(lap)

def pictureTest():
    #img=cv2.imread("picture/test01.jpg")#默认读取bgr格式
    #img_2 = colorRead("picture/test02.jpg")
    #img_1 = colorRead("picture/test01.jpg")
    img_2 = grayRead("picture/test02.jpg")
    img_1 = grayRead("picture/test01.jpg")
    #cat = cutpic(img)
    #img += 10#add函数越界直接255，+则取余
    
    img_2 = cv2.resize(img_2,(1080,730))
    #res = cv2.addWeighted(img_1,0.5,img_2,0.5,0)#融合
    #res_copy = thre(res)#填充边框
    #img_1 = smooth(img_1)#平滑处理
    img_1 = sobel(img_1)
    showInfo(img_1)
    showImg(0,img_1,"img")
    


def movTest():
    vc =  cv2.VideoCapture("mp4/test01.mp4")
    if(vc.isOpened()):
        open,frame = vc.read()#一帧一帧读取图像
    else:
        open = False
    while open:
        ret,frame = vc.read()
        if(frame is None):
            break
        if(ret == True):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("result",gray)
            if(cv2.waitKey(1) & 0xFF == 27):
                break
    vc.release()
    cv2.destoryAllWindows()


#movTest()
pictureTest()
