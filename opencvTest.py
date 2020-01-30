import cv2
import numpy as np
import matplotlib.pyplot as plt


method = ["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]

def matchT(img1,img2):
    for meth in method:
        img_copy = img1.copy()
        methodItem = eval(meth)
        res = cv2.matchTemplate(img1,img2,methodItem)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
        print(res.shape)
        print(min_val)
        print(max_val)
        print(min_loc)
        print(max_loc)#根据方法不同取值
        print(img2.shape)
        if(methodItem in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]):
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0]+img2.shape[1],top_left[1]+img2.shape[0])
        cv2.rectangle(img_copy,top_left,bottom_right,255,2)
    
        plt.subplot(121),plt.imshow(res,cmap="gray")
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(img_copy,cmap="gray")
        plt.xticks([]),plt.yticks([])
        plt.suptitle(meth)
        plt.show()
    



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

def canny(img):
    v1 = cv2.Canny(img,140,150)#80 150 双重阈值
    v2 = cv2.Canny(img,50,100)
    res = np.hstack((v1,v2))
    return res

def Lapras(img):
    return (img - cv2.pyrUp(cv2.pyrDown(img)))

def up(img ,upordown):
    if(upordown ==1):
        return cv2.pyrUp(img)
    else:
        return cv2.pyrDown(img)

def thre(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) #转成二值图像
    binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#binary 原图像 contours轮廓信息，hierarchy层级
    copy_img = img.copy()#会改原图
    #
    app = approximate(contours)
    res = cv2.drawContours(copy_img,[app],-1,(0,0,255),2)#-1指所有轮廓，第几层级的轮廓，轮廓颜色,线条宽度
    showImg(0,img,"img")
    showImg(0,res,"img")
    return res,img,contours

def conInfo(contours):
    cnt = contours[0]
    print(cv2.arcLength(cnt,True))
    print(cv2.contourArea(cnt))
   
     
def approximate(contours):
    #epsilon = 0.1*cv2.arcLength(contours,True)#一般是周长百分比
    epsilon = 0.1
    approx = cv2.approxPolyDP(contours[30],epsilon,True)
    return approx

def cal(img):
    return cv2.calcHist([img],[0],None,[256],[0,256])
#直方图的均衡化
def equ(img):
    return cv2.equalizeHist(img)
#自适应均衡化 噪音点影响结果，分区域的均衡化可能会出现边界
def cequ(img):
    return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(img)
def mydft(img_float32):
    dft = cv2.dft(img_float32,flags = cv2.DFT_COMPLEX_OUTPUT)#傅里叶变换
    dft_shift = np.fft.fftshift(dft)#低频放到中心位置
    magnitude = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))#将频率映射到0-255
    return magnitude

def lowpass(img):
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32,flags = cv2.DFT_COMPLEX_OUTPUT)#傅里叶变换
    dft_shift = np.fft.fftshift(dft)#低频放到中心位置
    rows,cols = img.shape
    crow,ccol = int(rows/2),int(cols/2)

    #构造滤波器，就是个掩码,低通,高通就把0和1反过来
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30,ccol-30:ccol+30] = 1


    fshift = dft_shift*mask#抠图
    f_ishift = np.fft.ifftshift(fshift)#恢复原位
    img_back = cv2.idft(f_ishift)#傅里叶逆变换
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])#实部虚部处理
    
    #显示部分
    plt.subplot(121),plt.imshow(img,cmap="gray")
    plt.title("input"),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img_back,cmap="gray")
    plt.title("res"),plt.xticks([]),plt.yticks([])

    plt.show()
def pictureTest():
    #img=cv2.imread("picture/test01.jpg")#默认读取bgr格式
    #img_2 = colorRead("picture/test05.jpg")
    #img_1 = colorRead("picture/test01.jpg")
    img_2 = grayRead("picture/test05.jpg")
    img_1 = grayRead("picture/test01.jpg")
    #cat = cutpic(img)
    #img += 10#add函数越界直接255，+则取余
    
    #img_2 = cv2.resize(img_2,(1080,730))
    #res = cv2.addWeighted(img_1,0.5,img_2,0.5,0)#融合
    #res_copy = thre(res)#填充边框
    #img_1 = smooth(img_1)#平滑处理
    #img_1 = sobel(img_1)
    #img_1 = canny(img_1)
    #img_1 = up(img_1,2)
    #img_1 = Lapras(img_1)
    
    #img,img_1,contours = thre(img_1)
    #conInfo(contours)
    
    #showImg(0,img_1,"img")
    #showImg(0,img,"img")
    #img = cequ(img_1)
    #showInfo(img)
    #mag = mydft(np.float32(img_1))
    #plt.subplot(121),plt.imshow(img_1,cmap="gray")
    #plt.title("input"),plt.xticks([]),plt.yticks([])
    #plt.subplot(122),plt.imshow(mag,cmap="gray")
    #plt.title("magnitude spectrum"),plt.xticks([]),plt.yticks([])
    #plt.hist(img.ravel(),256)#直方图
    
    #plt.plot(img,color='b')#折线图 
    #plt.xlim([0,256])
    #plt.show()
    #showImg(0,img_1,"img")
    #showImg(0,img,"img")
    #matchT(img_1,img_2)

    lowpass(img_1)
    


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
