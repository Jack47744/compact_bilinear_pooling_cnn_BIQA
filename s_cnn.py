import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

class S_CNN(ImageClassificationBase):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size=3, padding=1, stride = 1)
        self.conv1_bn=nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size=3, padding=1, stride = 2)
        self.conv2_bn=nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size=3, padding=1, stride = 1)
        self.conv3_bn=nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1, stride = 2)
        self.conv4_bn=nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1, stride = 1)
        self.conv5_bn=nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1, stride = 2)
        self.conv6_bn=nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1, stride = 1)
        self.conv7_bn=nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, padding=1, stride = 1)
        self.conv8_bn=nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, padding=1, stride = 2)
        self.conv9_bn=nn.BatchNorm2d(128)
        

        # average pooling (kernel_size, stride)
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

        # fully conected layers:
        self.fc1 = nn.Linear(14*14*128, 128)
        self.fc1_bn=nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_bn=nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)
        self.fc3_bn=nn.BatchNorm1d(n_classes)

    def forward(self, x, training=True):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        x = self.conv4(x)
        x = F.relu(self.conv4_bn(x))
        x = self.conv5(x)
        x = F.relu(self.conv5_bn(x))
        x = self.conv6(x)
        x = F.relu(self.conv6_bn(x))
        x = self.conv7(x)
        x = F.relu(self.conv7_bn(x))
        x = self.conv8(x)
        x = F.relu(self.conv8_bn(x))
        x = self.conv9(x)
        x = F.relu(self.conv9_bn(x))
        x = self.pool(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        x = F.dropout(x, 0.2)

        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = F.dropout(x, 0.2)

        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        x = F.dropout(x, 0.2)

        return x