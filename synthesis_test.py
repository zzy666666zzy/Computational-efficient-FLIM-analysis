# -*- coding: utf-8 -*-
#ZZY 02/June/2021

import os
import torch
import numpy as np
import scipy.io as sio
from S_TauNet_AVE_model import S_TauNet_AVE
from Q_S_TauNet_AVE_model import Quan_S_TauNet_AVE
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from torchsummary import summary
from thop import profile
from sklearn.metrics import mean_squared_error
import scipy.io as io
from S_TauNet_AVE_model_logscaling import S_TauNet_AVE_loghist

start = time.time()

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave'
sample_file='Synthetic_2D_Pattern_100to1000'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('Compressed_Hist')  
# hist_log_scaling=DataSet.get('Compressed_Hist') 
tau2d_amp_ave_gt=DataSet.get('tau2d_amp_ave')  
tau2d_inten_ave_gt=DataSet.get('tau2d_inten_ave')  

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])

hist = Variable(torch.from_numpy(hist))

#use gpu
# PATH=r'./addernet_pth/good_model_loss_0.018485_bi_40000_mono100_mixed_Np10to100.pth'
# model = S_TauNet_AVE()
#use cpu
#checkpoint = torch.load(PATH, map_location='cpu')
#model.load_state_dict(checkpoint)

# PATH=r'./QAdderNet_loss_0.021396.pth'
# model = Quan_S_TauNet_AVE()

PATH='S_TauNet_AVE_train_log_scaling.py_val_loss_0.038527.pth'
model = S_TauNet_AVE_loghist()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
hist=torch.unsqueeze(hist,0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
hist = hist.to(device)

tau_amp=np.zeros(65536)
tau_inten=np.zeros(65536)
threshold=0

hist_nothre=hist.unsqueeze(2)
with torch.no_grad():
    for i in range(65536):
        hist_thre=torch.squeeze(torch.squeeze(hist))
        if torch.sum(hist_thre[i,:],dim=0)<threshold:
            yhat=0
        else:
            hist_adder = hist_nothre[:,i,:,:,:]/torch.max(hist_nothre[:,i,:,:,:])
            yhat = model(hist_adder)
            tau_amp[i]=yhat[0]
            tau_inten[i]=yhat[1]
        if (i%10000==0):
            print('{} pixel processed'.format(i))

tau_amp=tau_amp.reshape(256,256)
tau_inten=tau_inten.reshape(256,256)

tau_amp[np.isnan(tau_amp)] = 0
tau_inten[np.isnan(tau_inten)] = 0

end = time.time()
#%%
print('------Inference Done------')
print(f"Runtime of the program is {end - start}")

sio.savemat(r"./mat_result/adder_net_log_scale_100to1000_amp.mat", {"adder_net_log_scale_100to1000_amp":tau_amp})
sio.savemat(r"./mat_result/adder_net_log_scale_100to1000_inten.mat", {"adder_net_log_scale_100to1000_inten":tau_inten})
#%%
tau_amp=np.transpose(tau_amp,[1,0])
plt.imshow(tau_amp, cmap='gist_ncar')
plt.colorbar()
plt.clim(0, 3)
plt.axis('off')
plt.show()
#%%
tau_inten=np.transpose(tau_inten,[1,0])
plt.imshow(tau_inten, cmap='gist_ncar')
plt.colorbar()
plt.clim(0, 3)
plt.axis('off')
plt.show()
#%%
mse_amp=mean_squared_error(tau_amp,tau2d_amp_ave_gt)
mse_inten=mean_squared_error(tau_inten,tau2d_inten_ave_gt)
# input = torch.randn(1,1,1,256).cuda()
# macs, params = profile(model, inputs=(input, ))
# summary(model,input)

