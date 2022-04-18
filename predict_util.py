from dataset_util import *
from cbp_cnn import *
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time

import logging


class PREDICT_UTIL:
    def __init__(self, model_path: str, s_cnn_path: str):
        """The inference utility class that provides the prediction of the given image

        Args:
            model_path (str): The path of the model
            s_cnn_path (str): The path of the SCNN model
        """
        self.model_path = model_path
        self.s_cnn_path = s_cnn_path
        self.device = get_default_device()
        self.invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                 std=[1/0.229, 1/0.224, 1/0.225]),
                                            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                 std=[1., 1., 1.]),
                                            ])
        logger = logging.getLogger('PREDICT_UTIL')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False

        self.logger = logger

    def load_model(self):
        """Loading the model into memory
        """
        try:
            start_time = time.time()
            self.logger.info(f'start loading model')
            self.model = DB_CNN(device=self.device, options=False,
                                model_path=self.s_cnn_path).to(self.device)
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=torch.device(self.device)))
            self.logger.info(
                f'Load model complete using {time.time()-start_time} seconds')
        except:
            self.logger.warning('Cannot load model')

    def gen_skin_img(self, img, crop_shape=(448, 448)):

        width, height = img.size
        crop_y = crop_shape[0]
        crop_x = crop_shape[1]
        img_list = []
        if height <= 1.2*crop_y or width <= 1.2*crop_x:
            transform_test2 = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop((448, 448)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
            return transform_test2(img)
        elif height > width:
            new_width = 1*crop_x
            new_heigth = (new_width/width)*height
            # print(new_width, new_heigth)
            transform_test2 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    size=(int(new_heigth), int(new_width))),
                torchvision.transforms.CenterCrop((448, 448)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
            return transform_test2(img)
        else:
            new_height = 1*crop_y
            new_width = (new_height/height)*width
            transform_test2 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    size=(int(new_height), int(new_width))),
                torchvision.transforms.CenterCrop((448, 448)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
            return transform_test2(img)

    def predict_img(self, IMG_PATH):
        with torch.no_grad():
            self.model.eval()
            img = self.gen_skin_img(Image.open(IMG_PATH).convert('RGB'))

            torch.cuda.empty_cache()
            xb = to_device(img.unsqueeze(0), self.device)
            yb = self.model(xb)
            # print(yb.item())
            npimg = self.invTrans(img).cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            return yb.item()

    def predict_img_2(self, img):
        start_time = time.time()
        self.logger.info('Start predict')
        with torch.no_grad():
            self.model.eval()
            torch.cuda.empty_cache()
            img = self.gen_skin_img(img)
            xb = to_device(img.unsqueeze(0), self.device)
            yb = self.model(xb)
            self.logger.info(
                f'Predict finish using {time.time() - start_time} seconds')
            image_score = yb.item()
            self.logger.info(f'Image score is {image_score}')
            return image_score
