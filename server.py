from fastapi import FastAPI, File, UploadFile
import uvicorn
from predict_util import *
import urllib.request
import ssl
from io import BytesIO

ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://www.python.org')

S_CNN_PATH = 'revised_model_13_10_2021_0.pth'
MODEL_PATH = 'db_cnn_v2_challenge.pth'
Predict = PREDICT_UTIL(model_path=MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()


app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = Predict.predict_img_2(image)
    # prediction = predict(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
