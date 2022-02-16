from predict_util import *
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://www.python.org')
# print(response.read().decode('utf-8'))
S_CNN_PATH = 'revised_model_13_10_2021_0.pth'
MODEL_PATH = 'db_cnn_v2_challenge.pth'
EXAMPLE_IMG_PATH = '/Users/metis_sotangkur/Desktop/istockphoto-1295274245-170667a.jpeg'
Predict = PREDICT_UTIL(model_path=MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()
print(Predict.predict_img(EXAMPLE_IMG_PATH))
# print(torch.__version__)