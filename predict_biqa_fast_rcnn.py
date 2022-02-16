from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
from PIL import Image
from predict_util import *
import urllib.request
import os
import ssl
def filter(result, threshold):
    '''
    parameters:
    result - array of size (1,N,5)
    threshold - score threshold
    return:
    list of predictions that score >= threshold {"coordinates":list of 4 coordinates, "score":score} 
    '''
    result = np.array(result).reshape(-1, 5)
    preds = []
    for p in result:
        if p[4] >= threshold:
            preds.append({"coordinates": p[0:4].tolist(), "score": p[4]})

    return preds


MODEL_CONFIG = "./Object_localization_melanoma-main/model_config_SmoothL1.py"
CHECKPOINT_PTH = "./Object_localization_melanoma-main/mask_rcnn_smoothl1.pth"
IMG_PTH = "./Object_localization_melanoma-main/example.jpg"
mask_rcnn_model = init_detector(MODEL_CONFIG, CHECKPOINT_PTH, device='cpu')
result = inference_detector(mask_rcnn_model, IMG_PTH)
predicted_result = filter(result, 0.3)
im = Image.open(IMG_PTH)
print('complete cropping suspected area(s)')
crop_img_path_list = []


ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://www.python.org')
# print(response.read().decode('utf-8'))
S_CNN_PATH = 'revised_model_13_10_2021_0.pth'
MODEL_PATH = 'db_cnn_v2_challenge.pth'
EXAMPLE_IMG_PATH = '/Users/metis_sotangkur/Desktop/istockphoto-1295274245-170667a.jpeg'
Predict = PREDICT_UTIL(model_path=MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()
print('complete loading model')


score_list = []
for index, roi in enumerate(predicted_result):
    coordinate = tuple(roi['coordinates'])
    cropped = im.crop(coordinate)
    img_crop_path = f'./crop_img/crop_no_{index}.jpg'
    crop_img_path_list.append(img_crop_path)
    cropped.save(img_crop_path)
    area_score = Predict.predict_img(img_crop_path)
    score_list.append(area_score)
    #cropped = im.crop((1,2,300,300))
print(score_list)
mydir = 'crop_img'
filelist = [ f for f in os.listdir(mydir) if f.endswith(".jpg") ]

print(Predict.predict_img(IMG_PTH))

for f in filelist:
    os.remove(os.path.join(mydir, f))
#show_result_pyplot(mask_rcnn_model, IMG_PTH, result, score_thr=0.3)
 