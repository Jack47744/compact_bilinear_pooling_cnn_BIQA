import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from s_cnn import *
from compact_bilinear_pooling import *

class ImageRegression(nn.Module):
    """
    A based class which will be used for creating the Compact Bilinear Pooling CNN

    Methods
    -------
    training_step(batch)
        
    """
    def training_step(self, batch):
        images, mos_scores = batch 
        out = self(images)                  # Generate predictions
        mse_loss = nn.MSELoss()
        loss = mse_loss(out[:, 0].float(), mos_scores.float())  # Calculate loss
        
        # print('loss:',loss)
        return loss
    
    def validation_step(self, batch):
        images, mos_scores = batch
        out = self(images)                    # Generate predictions
        mse_loss = nn.MSELoss()
        loss = mse_loss(out[:, 0].float(), mos_scores.float())   # Calculate loss
        # print(out[:, 0].float(), mos_scores.float())
        
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        # batch_srccs = [x['val_srcc'] for x in outputs]
        # epoch_acc = torch.stack(batch_srccs).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], srcc_train: {:.5f}, plcc_train: {:.5f}, srcc_val: {:.5f}, plcc_val: {:.5f}, loss_train: {:.4f}, loss_test: {:.4f}".format(
            epoch, result['srcc_train'], result['plcc_train'], result['srcc_val'], result['plcc_val'], result['train_loss'], result['val_loss']))

class DB_CNN(ImageRegression):
    def __init__(self, device, options=True, dropout_ratio=0.2, model_path="revised_model_13_10_2021_0.pth"):
        super().__init__()
        s_cnn = S_CNN(3, 27).to(device)
        # s_cnn.load_state_dict(torch.load(model_path))
        s_cnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        self.synth_extractor = torch.nn.Sequential(*(list(s_cnn.children())[:-7]))
        self.auth_extractor = torch.nn.Sequential(*(list(vgg16.children())[0][:-1]))
        
        self.compact_bilinear_pooling = CompactBilinearPooling(128*28*28, 512*28*28, 128*512).to(device)
        self.flatten = torch.nn.Flatten()
        self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc = torch.nn.Linear(512*128, 1)
        self.bn = nn.BatchNorm1d(512*128)
        if options == True:
            # Freeze all previous layers.
            for param in self.synth_extractor.parameters():
                param.requires_grad = False
            for param in self.auth_extractor.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x, training=True):
        synth_result = self.synth_extractor(x)
        auth_result = self.auth_extractor(x)
        synth_result = self.flatten(synth_result)
        auth_result = self.flatten(auth_result)
        bi_result = self.compact_bilinear_pooling(synth_result, auth_result)
        # x = torch.sqrt(bi_result + 1e-8)
        x = torch.nn.functional.normalize(bi_result)
        # print(x.shape)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
