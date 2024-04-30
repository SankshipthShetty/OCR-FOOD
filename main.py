# import pandas as pd
# import numpy as np
# from glob import glob
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from PIL import Image
#
# #pytesseract
# #easyocr
# #keras_ocr
#
#
# #------------------
# #open image
# import cv2
# image_file="C:/Users/sanks/OneDrive/Pictures/Screenshots/data1.jpg"
# img=cv2.imread(image_file)
# # cv2.imshow("original image",img)
# # cv2.waitKey(0)
# #
#
# def display(im_path):
#
#     dpi=80
#     # im_data=plt.imread(im_path)
#     # print(im_data.shape)
#     # height,width,depth=im_data.shape
#     #
#     # figsize=width/float(dpi),height/float(dpi)
#     #
#     # fig=plt.figure(figsize=figsize)
#     # ax=fig.add_axes([0,0,1,1])
#     #
#     # ax.axis('off')
#     #
#     # ax.imshow(im_data,cmap='gray')
#     #
#     # plt.show()
#     im_data = plt.imread(im_path)
#     # Check if the image has three dimensions (height, width, depth) or two dimensions (grayscale)
#     if len(im_data.shape) == 3:
#         height, width, depth = im_data.shape
#     else:
#         height, width = im_data.shape
#         depth = 1  # Assuming it's grayscale, so depth is 1
#
#     # Calculate figure size
#     figsize = width / float(dpi), height / float(dpi)
#
#     # Create figure and axes
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis('off')
#
#     # Display the image
#     if depth == 3:
#         ax.imshow(im_data)
#     else:
#         ax.imshow(im_data, cmap='gray')
#
#     plt.show()
#
#
# #display(image_file)
#
#
# #inverted image
# # inverted_image=cv2.bitwise_not(img)
# # cv2.imwrite("D:/Projects/Computer Vision/OCR(FOOD)/inverted.jpg",inverted_image)
# # display("D:/Projects/Computer Vision/OCR(FOOD)/inverted.jpg")
#
# #rescaling
#
#
# #binarization
# def grayscale(image):
#     return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #
# gray_image=grayscale(img)
# cv2.imwrite("D:/Projects/Computer Vision/OCR(FOOD)/binarization.jpg",gray_image)
#
# # display("D:/Projects/Computer Vision/OCR(FOOD)/binarization.jpg")
#
# #
# thresh,im_bw=cv2.threshold(gray_image,100,230,cv2.THRESH_BINARY)
# # cv2.imwrite("D:/Projects/Computer Vision/OCR(FOOD)/thresh.jpg",im_bw)
# # display("D:/Projects/Computer Vision/OCR(FOOD)/thresh.jpg")
#
#
# #noise
# import numpy as np
# def noice_removal(image):
#
#     kernel=np.ones((1,1),np.uint8)
#     image=cv2.dilate(img,kernel,iterations=1)
#     kernel= np.ones((1,1),np.uint8)
#     image=cv2.erode(image,kernel,iterations=1)
#     image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
#     image=cv2.medianBlur(image,3)
#     return image
#
# no_noise=noice_removal(im_bw)
# cv2.imwrite("D:/Projects/Computer Vision/OCR(FOOD)/noise.jpg",no_noise)
#
# display("D:/Projects/Computer Vision/OCR(FOOD)/noise.jpg")


import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'D:/Setups/Tesseract/tesseract.exe'
imgfile="C:/Users/sanks/OneDrive/Pictures/Screenshots/data1.jpg"

no_noise="D:/Projects/Computer Vision/OCR(FOOD)/thresh.jpg"

img=Image.open(no_noise)

ocr_result=pytesseract.image_to_string(img)

print(ocr_result)

