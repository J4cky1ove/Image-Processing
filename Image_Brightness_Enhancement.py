import numpy as np
import  cv2
import time
import os
def ALTM(img):
    h, w = img.shape[:2]
    res = np.float32(img)  # res = np.array(img, dtype=np.float32)  # 转换为32位图像
    Lwmax = res.max()
    log_Lw = np.log(0.001 + res)
    Lw_sum = log_Lw.sum()
    Lwaver = np.exp(Lw_sum / (h * w))
    Lg = np.log(res / Lwaver + 1) / np.log(Lwmax / Lwaver + 1)
    res = Lg * 255.0  # 不使用分段线性增强
    dst = np.uint8(res)  # dst = cv2.convertScaleAbs(res)
    return dst
picturePath = './low'
outpicturePath='./Lightenhanced'
t=[]
for fi in os.listdir(picturePath):
    if fi.endswith(".jpg"):     #文件是以jpg结尾的
            I = cv2.imread(picturePath+'/'+fi) / 255.0
            time_start = time.time()
            res = np.zeros(I.shape)
            for k in range(3):
                res[:, :, k] = ALTM(I[:, :, k])
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            t.append(time_sum)
            print(time_sum)
            cv2.imwrite(outpicturePath+'/'+fi, res)
            print(fi)
print(np.mean(t))


