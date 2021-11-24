from predict_util import *
S_CNN_PATH = ''
MODEL_PATH = ''
EXAMPLE_IMG_PATH = ''
Predict = PREDICT_UTIL(model_path=MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()
print(Predict.predict_img(EXAMPLE_IMG_PATH))