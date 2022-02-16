import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""

        return len(self.dl)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.transform(Image.open(row["FilePath"])),
            row["img_score"],
        )

def get_data_loader(train, test, random_seed, batch_size, transform_train, transform_test):
    
    dataset_train = MyDataset(train, transform = transform_train)
    dataset_test = MyDataset(test, transform = transform_test)
    torch.manual_seed(random_seed)
    val_size = int(len(train)*0.2)
    train_size = len(dataset_train) - val_size
    train_ds, val_ds = random_split(dataset_train, [train_size, val_size])
    print('train size:', len(train_ds))
    print('validation size:', len(val_ds))
    batch_size = batch_size
    device = get_default_device()
    print(device)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    test_dl =  DataLoader(dataset_test, batch_size//2, num_workers=4, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    return train_dl, val_dl, test_dl

def get_data_loader2(train, test, random_seed, batch_size, transform_train, transform_test):
    dataset_train = MyDataset(train, transform = transform_train)
    dataset_test = MyDataset(test, transform = transform_test)
    torch.manual_seed(random_seed)
    batch_size = batch_size
    device = get_default_device()
    print(device)
    train_dl = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl =  DataLoader(dataset_test, batch_size, num_workers=4, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    return train_dl, test_dl
