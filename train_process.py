from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from dataset_util import *
from train_util import *
import torchvision
import pandas as pd

def get_train_test_meta(meta, train_size=0.80, random_state=2021):
    return train_test_split(meta, train_size=train_size, shuffle=True, random_state=random_state)
def get_data_set(data_set_name):
    path_tid2013 = '/content/drive/My Drive/image_quality_dataset/TID2013/'
    path_live = '/content/drive/My Drive/image_quality_dataset/LIVE/'
    path_live_md = '/content/drive/My Drive/image_quality_dataset/LIVE_MD/'
    path_live_challenge = '/content/drive/My Drive/image_quality_dataset/LIVE_CHALLENGE/'

    if data_set_name == 'LIVE':
        meta_live = pd.read_csv(path_live + 'meta.csv')
        meta_live['img_name'] = path_live + 'Train/' + meta_live['img_name']
        # print(meta_live['img_score'].mean())
        meta_live['source'] = 1
        meta_live = meta_live.rename(columns={'img_name': 'FilePath'})
        return meta_live
    
    elif data_set_name == 'LIVE_MD':
        meta_live_md = pd.read_csv(path_live_md + 'meta.csv')
        meta_live_md['img_name'] = meta_live_md['img_name'].str.slice(start=1, stop=-1)
        meta_live_md['img_name'] = path_live_md + 'Train/' + meta_live_md['img_name']
        # print(meta_live_md['img_score'].mean())
        meta_live_md['source'] = 2
        meta_live_md = meta_live_md.rename(columns={'img_name': 'FilePath'})
        return meta_live_md

    elif data_set_name == 'LIVE_CHALLENGE':
        meta_live_challenge = pd.read_csv(path_live_challenge + 'meta.csv')
        meta_live_challenge['img_name'] = meta_live_challenge['img_name'].str.slice(start=1, stop=-1)
        meta_live_challenge['img_name'] = path_live_challenge + 'Train/' + meta_live_challenge['img_name']
        # print(meta_live_challenge['img_score'].mean())
        meta_live_challenge['source'] = 3
        meta_live_challenge = meta_live_challenge.rename(columns={'img_name': 'FilePath'})
    
    elif data_set_name == 'TID2013':
        meta_tid = pd.read_csv(path_tid2013 + 'meta.csv').rename(columns={'mos_score': 'img_score'})
        meta_tid['img_name'] = path_tid2013 + 'Train/' + meta_tid['img_name']
        # print(meta_tid['img_score'].mean())
        meta_tid['source'] = 0
        meta_tid = meta_tid.rename(columns={'img_name': 'FilePath'})
        return meta_tid

if __name__ == "__main__":
    dataset = get_data_set('LIVE_CHALLENGE')
    live_cl_train, live_cl_test = get_train_test_meta(dataset.drop('source', 1))

    srcc_list = []
    plcc_list = []
    kf = KFold(n_splits=5, random_state=1234, shuffle=True)
    df = dataset.drop('source', 1)
    device = get_default_device()
    transform_train = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomCrop(size=(448, 448), pad_if_needed=True, padding_mode='reflect'),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))])
    transform_test = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.RandomCrop(size=(448, 448), pad_if_needed=True, padding_mode='reflect'),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))])
    device = get_default_device()
    for train_index, test_index in kf.split(df):
        # print(train_index, test_index)
        print(len(train_index)%8, len(test_index)%8)
        if len(train_index)% 8 == 1:
            train_index = train_index[:-1]
        print(len(train_index)%8)
        # if len(test_index)% 8 == 0:
        #     test_index = test_index[:-1]
        live_cl_train = df.iloc[train_index]
        live_cl_test = df.iloc[test_index]
        train_dl, test_dl = get_data_loader2(live_cl_train, live_cl_test, 1234, 8, transform_train, transform_test)
        SAVE_PATH = "/content/drive/My Drive/Image_data/db_cnn_v2_challenge_4.pth"

        history, model, model2 = train_process(freeze=False, epochs=30, max_lr=1e-4, grad_clip=0.1, weight_decay=5e-4, opt_func=torch.optim.Adam, train_dl=train_dl, val_dl=test_dl, device=device, save_path=SAVE_PATH)
        plot_losses(history)
        plot_srcc(history)
        plot_plcc(history)
        # test_srcc, test_plcc = get_result(model, live_cl_test, transform_test)
        test_srcc, test_plcc = get_result(model2, live_cl_test, transform_test, device)
        srcc_list.append(test_srcc)
        plcc_list.append(plcc_list)
        # break
