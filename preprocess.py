import numpy as np
import cv2

def background_remove(file, image_array):
    img0 = cv2.fastNlMeansDenoisingColored(image_array, None, 150, 150,7,21)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, ksize=(7,7), sigmaX=1, sigmaY=1)
    ret, imthres = cv2.threshold(img1, 60, 1, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    imgthres = cv2.dilate(imthres,k)
    cnts, _ = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(cnts[-1])
    cv2.drawContours(img1, cnts, -1, (255, 0, 0 ), 4)

    return image_array



def zero_padding(img):
    height, width = img.shape[0], img.shape[1]
    margin = [np.abs(height-width)//2, np.abs(height - width) //2]
    if np.abs(height-width)%2 !=0:
        margin[0] +=1
    if height < width:
        margin_list = [margin, [0,0]]
    else:
        margin_list = [[0,0], margin]
    if img.shape[2] == 3:
        margin_list.append([0,0])
    new_array = np.pad(img, margin_list, mode = "constant")
    return new_array


def Min_Max_Normalization(img):
    min_value = np.min(img)
    max_value = np.max(img)
    diviser = max_value - min_value
    if diviser ==0:
        if max_value !=0:
            new_array = img / max_value
        else:
            new_array = img
    else:
        new_array = (img - min_value)/(max_value-min_value)
    new_array = np.float32(new_array)
    return new_array


def Histogram_Equalization_CLAHE_Color(img, limit, kernel_size):
    img = make_unit8(img)
    ch1, ch2, ch3  = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(kernel_size, kernel_size))
    new_array = cv2.merge([clahe.apply(ch1),clahe.apply(ch2), clahe.apply(ch3)])
    return new_array

def make_unit8(array):
    ori_max = np.max(array)
    ori_min = np.min(array)
    diviser = ori_max - ori_min
    if diviser ==0:
        if ori_max !=0:
            array = np.uint8(array/ori_max*255)
        else:
            array = np.uint8(array)

    else:
        array = np.uint8((array-ori_min)/(diviser)*255)
    return array
