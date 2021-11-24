from dataset_util import *
from cbp_cnn import DB_CNN
import torchvision
class PREDICT_UTIL:
    def __init__(self, MODEL_PATH, S_CNN_PATH):
        self.model_path = MODEL_PATH
        self.s_cnn_path = S_CNN_PATH
    
    def load_model(self):
        self.device = get_default_device()
        self.model = DB_CNN(device=self.device, options=False, model_path=self.s_cnn_path).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))

    def gen_skin_img(self, img, crop_shape = (448, 448)):
    # print(img.size)
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
        # new_width = 1.2*crop_x
        # h/h2 = w/w2 -> h2 = (w2/w)*h
        # w2 = 1.2*crop_x
        # new_height = height 
            new_width = 1*crop_x
            new_heigth = (new_width/width)*height
            # print(new_width, new_heigth)
            transform_test2 = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(int(new_heigth), int(new_width))),                            
                    torchvision.transforms.CenterCrop((448, 448)),                                  
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))])
            return transform_test2(img)
        else:
            new_height = 1*crop_y
            new_width = (new_height/height)*width
            transform_test2 = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(int(new_height), int(new_width))),                             
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
            print(yb.item()) 
            return yb

        
    