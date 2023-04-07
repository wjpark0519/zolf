import numpy as np
import torch
import torch.nn as nn
import val

from model import ZolfModel
from data import MyDataset
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

batch_size = 10
epoch: int = 1000
learning_rate = 1e-4
val_iter = 20


PATH = './sample_y_sin.csv'
model_path = 'model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ZolfModel()
model.to(device)

train_csv = np.loadtxt(PATH, delimiter=',', dtype=float)
y_train_csv = train_csv[:int(len(train_csv)*9/10), :-2]
aod_train_csv = train_csv[:int(len(train_csv)*9/10), -2:-1]
aoa_train_csv =train_csv[:int(len(train_csv)*9/10), -1:]

y_train = torch.tensor(y_train_csv,dtype=torch.float32)
aod_train = torch.tensor(aod_train_csv,dtype=torch.float32)
aoa_train = torch.tensor(aoa_train_csv,dtype=torch.float32)

train_dataset = MyDataset(y_train, aod_train, aoa_train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    shuffle= True,
    drop_last= True,
)

y_val_csv = train_csv[int(len(train_csv)*9/10): , :-2]
aod_val_csv = train_csv[int(len(train_csv)*9/10): , -2:-1]
aoa_val_csv =train_csv[int(len(train_csv)*9/10): , -1:]

y_val = torch.tensor(y_val_csv,dtype=torch.float32)
aod_val = torch.tensor(aod_val_csv,dtype=torch.float32)
aoa_val = torch.tensor(aoa_val_csv,dtype=torch.float32)

val_dataset = MyDataset(y_val, aod_val, aoa_val)
val_dataloader = DataLoader(
    val_dataset,
    batch_size= batch_size,
    shuffle= True,
    drop_last= False,
)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(  optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)


loss_fn = nn.MSELoss(reduction = 'mean')

model.train()
for ep in range(epoch):
    for iter , (y, aod, aoa) in enumerate(train_dataloader):
        y = y.to(device)
        angle = torch.stack([aoa,aod], dim=1)
        angle = angle.reshape(-1,2)
        angle = angle.to(device)

        optimizer.zero_grad()
        est = model(y)
        # print(est)
        # print(angle)
        loss = loss_fn(est, angle)
        #print(loss)
        loss.backward()
        optimizer.step()
    scheduler.step()
    torch.save(model.state_dict(), model_path)

    if ep % val_iter ==0:
        loss = val.validation(val_dataloader, device, loss_fn, model_path)
        print('train epoch: ', ep, 'loss : ', loss)