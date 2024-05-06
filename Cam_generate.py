import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from DA_ConvLSTM import MineralCNNLSTM_Flatten
from GradCAM1D import GradCAM1D

# model = MineralCNN(num_classes=2)
# model.load_state_dict(torch.load('./cnn_model/mineral_an/mineral_cnn_lstm_model_epoch_40.pth'))
model = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1)
model.load_state_dict(torch.load('./cnn_lstm_model2/mineral_an/mineral_cnn_lstm_model_epoch_40.pth'))
# test data dir
folder_path = './plot'
all_files = os.listdir(folder_path)

all_data = []

temp_df = pd.read_csv('./mixed_spectra/sd_an_train_015s_5600.csv')

# 读取数据和文件名
file_names = []
column = temp_df.columns

for file in all_files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, names=column)
        df = df.iloc[1:]
        all_data.append(df)
        file_names.extend([file] * len(df))

test_data = pd.concat(all_data, ignore_index=True)
# test_data['en'] = 0

spectra_test = test_data.iloc[:, :-4].values
target_test = test_data.iloc[:, -4:].values.astype(np.float64)

input_tensor = torch.tensor(spectra_test[1], dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

grad_cam = GradCAM1D(model, 'conv3')
cam = grad_cam.generate_cam(input_tensor, 1)

original_size = input_tensor.shape[2]

# Use linear interpolation
upsampled_cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=original_size, mode='linear', align_corners=False).squeeze()

cam_numpy = upsampled_cam.cpu().detach().numpy()

activation_2d = np.expand_dims(cam_numpy, axis=0)

plt.figure(figsize=(12, 3))
plt.imshow(activation_2d, aspect='auto', cmap='jet')
plt.colorbar(label='Activation Intensity')
plt.yticks([])
plt.xlabel('Position')
plt.title('Grad-CAM++ 1D Activation as Barcode')

plt.savefig('./plot/cnn_lstm_model_with_confidence_an.svg', format='svg')

plt.show()