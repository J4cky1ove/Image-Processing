import numpy as np
import cv2
import os.path
import time

eps = np.finfo(np.double).eps  # 很小的非负数


def get_gauss_kernel(sigma, dim=2):
    ksize = int(sigma * 2) * 2 + 1
    x = np.linspace(-ksize // 2, ksize // 2, ksize)
    y = np.linspace(-ksize // 2, ksize // 2, ksize)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    if dim == 1:
        return kernel
    elif dim == 2:
        return kernel[:, None] * kernel


def gauss_blur(img, sigma):
    ksize = 7
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def simplest_color_balance(img_msrcr, s1, s2):
    sorted_img = np.sort(img_msrcr, axis=None)
    N = img_msrcr.size
    Vmin = sorted_img[int(N * s1)]
    Vmax = sorted_img[int(N * (1 - s2)) - 1]
    img_msrcr = np.clip(img_msrcr, Vmin, Vmax)
    img_msrcr = (img_msrcr - Vmin) * 255 / (Vmax - Vmin)
    return img_msrcr.astype('uint8')


def retinex_MSRCP(img,sigmas=[12,80,250],s1=0.01,s2=0.01):
    '''compare to others, simple and very fast'''
    Int=np.sum(img,axis=2)/3
    Diffs=[]
    for sigma in sigmas:
        Diffs.append(np.log(Int+1)-np.log(gauss_blur(Int,sigma)+1))
    MSR=sum(Diffs)/3
    Int1=simplest_color_balance(MSR,s1,s2)
    B=np.max(img,axis=2)
    A=np.min(np.stack((255/(B+eps),Int1/(Int+eps)),axis=2),axis=-1)
    # print("np.stack((255/(B+eps),Int1/(Int+eps)),axis=2)", np.stack((255/(B+eps),Int1/(Int+eps)),axis=2).shape)
    # print(A[...,None]*img)
    # print("A[...,None]", A[...,None].shape)
    # print("img", img.shape)

    blue_channel, green_channel, red_channel = cv2.split(img)
    #A = A[...,None].astype(np.uint8)
    red_result = A * red_channel
    green_result = A * green_channel
    blue_result = A * blue_channel
    result = cv2.merge([red_result, green_result, blue_result])
    print(green_result)
    return result.astype('uint8')

    # print(A[...,None]*img)
    # return (A[...,None]*img).astype('uint8')


if __name__ == '__main__':
    picturePath = './color_enhanced'
    outpicturePath = './color_enhanced_amsr'
    t = []
    for fi in os.listdir(picturePath):
        if fi.endswith(".jpg"):
            I = cv2.imread(os.path.join(picturePath, fi))
            print(fi)
            time_start = time.time()
            m = retinex_MSRCP(I)
            time_end = time.time()
            time_sum = time_end - time_start
            t.append(time_sum)
            print(time_sum)
            cv2.imwrite(os.path.join(outpicturePath, fi), m)
    print(np.mean(t))
