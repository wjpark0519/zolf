import numpy as np
import torch
import torch.nn as nn

from model import ZolfModel
from data import MyDataset
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def validation(
        dataloader,
        device,
        loss_fn,
        model_path
):
    model = ZolfModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for iter , (y, aod, aoa) in enumerate(dataloader):
            y = y.to(device)
            angle = torch.stack([aod,aoa], dim=1)
            angle = angle.reshape(-1,2)
            angle = angle.to(device)

            est = model(y)
            loss = loss_fn(est, angle)
            sub = est-angle
            # print(torch.cat([est, angle], dim=1))
            #print(angle)
            count_pi = 0
            count_the = 0
            for i in range(len(est)):
                if sub[i][0]<3 and sub[i][0]>-3 :
                    count_pi = count_pi + 1
                if sub[i][1] < 3 and sub[i][1] > -3 :
                    count_the = count_the + 1
            acc_pi = count_pi*100 / len(est)
            acc_the = count_the*100 / len(est)
            print('acc_pi: ', acc_pi,'%' ', acc_the: ', acc_the, '%')

    return loss

