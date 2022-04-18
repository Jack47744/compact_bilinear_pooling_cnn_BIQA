from fastapi import FastAPI, File, UploadFile
import uvicorn
from predict_util import *
from io import BytesIO
import logging
from fastapi.middleware.cors import CORSMiddleware

S_CNN_PATH = 'revised_model_13_10_2021_0.pth'
MODEL_PATH = 'db_cnn_v2_challenge.pth'
Predict = PREDICT_UTIL(model_path=MODEL_PATH, s_cnn_path=S_CNN_PATH)
Predict.load_model()

app = FastAPI()

# CONFIG ORIGIN SERVERS HERE
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("BIQA_SERVER")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False


def read_imagefile(file) -> Image.Image:
    logger.info('Read image')
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    logger.info(f'Receive {file.filename}')
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image = read_imagefile(await file.read())
    prediction = Predict.predict_img_2(image)
    # prediction = predict(image)
    response = {
        'quality_score': prediction,
        'img_name': file.filename
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
