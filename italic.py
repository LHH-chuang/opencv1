import cv2
import numpy as np

def italic(img):
    gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    # cvtColor（src，type）
    # src原图像，type转换类型
    # cv2.COLOR_BGR2BGRA 将alpha通道增加到BGR或者RGB图像中
    ret,binary = cv2.threshold(gray1,130,255,cv2.THRESH_BINARY)
    # threshold(src,thresh,maxval,type)
    # src原图像，thresh阈值，maxval输出图像的最大值，type阈值类型
    # THRESH_BINARY---二值阈值化
    kernel = np.ones((60,60), np.uint8)
    # 设置方框大小及类型
    dst=cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel)
    # cv2.morphologyEx(src, type, kernel)
    # src 原图像 type 运算类型 kernel 结构元素
    # cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
    # 开运算(open)：先腐蚀后膨胀的过程。
    kernel = np.ones((100,100), np.uint8)
    dst=cv2.morphologyEx(dst, cv2.MORPH_CLOSE,kernel)
    # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
    # 闭运算(close)：先膨胀后腐蚀的过程。
    gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    # BGR和灰度图的转换使用 cv2.COLOR_BGR2GRAY
    contours, hierarchy = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours(image, mode, method)
    # image一个8位单通道二值图像（非0即1） mode轮廓的检索模式:cv2.RETR_EXTERNAL表示只检测外轮廓  method为轮廓的近似办法: cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    # contours返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
    # hierarchy返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
    cnt = contours[0]
    # 选择contours列表中的第一个元素
    leftmost=tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost=tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
    #获得轮廓的最上面，最下面，最左边，最右边的点。
    x = bottommost[1]-topmost[1]
    #得到椭圆的短直径
    r = rightmost[0]-leftmost[0]
    #得到椭圆的长直径
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # 获得轮廓的重心
    pts1 = np.float32([[cX-2*x,cY+x],[cX+2*x,cY+x],[cX-2*r,cY+x+4*r],[cX+2*r,cY+x+4*r]])
    #从原图中获得需要进行透视变换的图像
    pts2 = np.float32([[0,0],[4*r,0],[0,4*r],[4*r,4*r]])
    #设置透视变换后输出图像的格式
    MM = cv2.getPerspectiveTransform(pts1,pts2)
    #cv2.getPerspectiveTransform(src,dst) 计算转换矩阵
    #src输入图像的，dst输出图像
    dst = cv2.warpPerspective(img,MM,(4*r,4*r))
    #warpPerspective进行透视变换
    return dst

#img = cv2.imread("coin10.jpg",cv2.IMREAD_GRAYSCALE)
#dst = italic(img)
#cv2.imwrite("img2.jpg",dst)