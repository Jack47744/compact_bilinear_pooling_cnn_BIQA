import torch
from scipy import stats
import torch.nn as nn
import matplotlib.pyplot as plt
from cbp_cnn import *
from dataset_util import *

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_correlation(model, data_loader):
    model.eval()
    mos_score_list = []
    model_score_list = []
    with torch.no_grad():
        for batch in data_loader:
            images, mos_scores = batch
            images = torch.tensor(images.cuda())

            model_score = model(images)[:, 0]
            mos_scores = torch.tensor(mos_scores.cuda())
            
            model_score_list = model_score_list +  model_score.cpu().tolist()
            mos_score_list = mos_score_list + mos_scores.cpu().tolist()
    srcc, _ = stats.spearmanr(mos_score_list,model_score_list)
    plcc, _ = stats.pearsonr(mos_score_list,model_score_list)
    return srcc, plcc
      

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, save_path='/content/drive/My Drive/Image_data/db_cnn_v2_challenge.pth'):
    best_plcc = 0.7
    torch.cuda.empty_cache()
    history = []
    print("model will be saved to: {}".format(save_path))
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        srcc_train, plcc_train = get_correlation(model, train_loader)
        srcc_val, plcc_val = get_correlation(model, val_loader)
        if plcc_val > best_plcc:
            torch.save(model.state_dict(), save_path)
            best_plcc = plcc_val
        result['srcc_train'] = srcc_train
        result['plcc_train'] = plcc_train
        result['srcc_val'] = srcc_val
        result['plcc_val'] = plcc_val
        model.epoch_end(epoch, result)
        history.append(result)
        # print("srcc_train: {:.5f}, plcc_train: {:.5f}, srcc_val: {:.5f}, plcc_val: {:.5f}".format(srcc_train, plcc_train, srcc_val, plcc_val))
    return history, save_path

def train_process(freeze, epochs, max_lr, grad_clip, weight_decay, opt_func, train_dl, val_dl, device):
    model = DB_CNN(options=freeze).to(device)
    history = [evaluate(model, val_dl)]
    print(history)
    histoty_tmp, save_path = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                                           grad_clip=grad_clip, 
                                           weight_decay=weight_decay, 
                                           opt_func=opt_func)
    history += histoty_tmp
    model2 = DB_CNN(options=freeze, device=device).to(device)
    model2.load_state_dict(torch.load(save_path))
    return history, model, model2

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def plot_srcc(history):
    train_srcc = [x.get('srcc_train') for x in history]
    val_srcc = [x.get('srcc_val') for x in history]
    plt.plot(train_srcc, '-bx')
    plt.plot(val_srcc, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('srcc')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def plot_plcc(history):
    train_plcc = [x.get('plcc_train') for x in history]
    val_plcc = [x.get('plcc_val') for x in history]
    plt.plot(train_plcc, '-bx')
    plt.plot(val_plcc, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('plcc')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def get_result(model, test, transform, device):
    model.eval()
    real = []
    predict = []
    dataset_test = MyDataset(test, transform)
    for img, score in dataset_test:
        torch.cuda.empty_cache()
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        real.append(score)
        predict.append(yb.item())
    plt.scatter(real, predict)
    plt.xlabel("real_MOS_score")
    plt.ylabel("model_MOS_score")
    test_srcc, _ = stats.spearmanr(real,predict)
    test_plcc, _ = stats.pearsonr(real,predict)
    print('srcc:', test_srcc, 'plcc:', test_plcc)
    plt.show()
    return test_srcc, test_plcc
