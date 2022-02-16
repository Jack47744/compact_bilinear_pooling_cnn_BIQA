import skimage.io
import skimage.filters
import sys
import cv2
import numpy as np
from PIL import Image
import PIL

from predict_util import *
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://www.python.org')
# print(response.read().decode('utf-8'))
S_CNN_PATH = 'revised_model_13_10_2021_0.pth'
MODEL_PATH = 'db_cnn_v2_challenge.pth'
Predict = PREDICT_UTIL(model_path = MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()


distorted_path = '/Users/metis_sotangkur/Desktop/Senior/Capstone/distorted_img/'
img_score = []
for image_num in range(12):
    tmp = []
    for level in range(10):
        if image_num < 10:
            image_num_str = f'0{image_num}'
        else:
            image_num_str = f'{image_num}'
        if level < 10:
            level_str = f'0{level}'
        else:
            level_str = f'{level}'
        img_name = f'{image_num_str}_{level_str}.jpg'
        img_path = distorted_path + img_name
        image = Image.open(img_path).convert('RGB')
        score = Predict.predict_img_2(image)
        tmp.append(score)
    img_score.append(tmp)
print(img_score)