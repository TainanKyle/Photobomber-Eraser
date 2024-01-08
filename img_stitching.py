import numpy as np
import matplotlib.pyplot as plt
import cv2 # cv2 version-> 3.4.2 (The version is important for using SIFT)
import random # using for RANSAC algorithm
from classes import Stitcher, Homography, Blender

fileNameList = [('test_1_color', 'test_2_color')]
src_path = "./result/mask/"
saveFilePath = "./u4.jpg"
adjust_ratio = 0.2

def adjust(img):
    (h, w) = img.shape[:2]
    for i in range(1, h):
        for j in range(1, w):
            img[i, j] = (img[i-1, j] + img[i, j-1]) * adjust_ratio / 2 + img[i, j] * (1 - adjust_ratio)
    
    return img

def stitching(img_left, img_right, saveFilePath):
    
    # The stitch object to stitch the image
    blending_mode = "linearBlending" # three mode - noBlending、linearBlending、linearBlendingWithConstant
    stitcher = Stitcher()
    warp_img = stitcher.stitch([img_left, img_right], blending_mode)

    # warp_img = adjust(warp_img)

    # # plot the stitched image
    # plt.figure(13)
    # plt.title("warp_img")
    # plt.imshow(warp_img[:,:,::-1].astype(int))

    # save the stitched iamge
    # saveFilePath = "./u4.jpg".format(fileName1, fileName2, blending_mode)
    cv2.imwrite(saveFilePath, warp_img)
    print("Result photo saved in", saveFilePath)
        
if __name__ == "__main__":
    for fname1, fname2 in fileNameList:
        # Read the img file
        fileName1 = fname1
        fileName2 = fname2
        img_left = cv2.imread(src_path + fileName1 + ".png")
        img_right = cv2.imread(src_path + fileName2 + ".png")
    
    stitching(img_left, img_right, saveFilePath)