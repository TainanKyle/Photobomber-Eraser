import numpy as np
import matplotlib.pyplot as plt
import cv2 # cv2 version-> 3.4.2 (The version is important for using SIFT)
import random # using for RANSAC algorithm
from classes import Stitcher, Homography, Blender


if __name__ == "__main__":
    
    fileNameList = [('test_1', 'test_2')]
    for fname1, fname2 in fileNameList:
        # Read the img file
        src_path = "./test/"
        fileName1 = fname1
        fileName2 = fname2
        img_left = cv2.imread(src_path + fileName1 + ".jpg")
        img_right = cv2.imread(src_path + fileName2 + ".jpg")
        
        # The stitch object to stitch the image
        blending_mode = "linearBlending" # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitcher()
        warp_img = stitcher.stitch([img_left, img_right], blending_mode)

        # # plot the stitched image
        # plt.figure(13)
        # plt.title("warp_img")
        # plt.imshow(warp_img[:,:,::-1].astype(int))

        # save the stitched iamge
        saveFilePath = "./u4.jpg".format(fileName1, fileName2, blending_mode)
        cv2.imwrite(saveFilePath, warp_img)