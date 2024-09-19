import Datasets
import Resnet
import NN_init
import pandas as pd
import NN_training
import torch
import torch.utils.data as data
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras import utils
import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dannye
ee = pd.read_csv(filepath_or_buffer=NN_init.finename,
                         names=['title', 'class'], encoding='Windows-1251', delimiter=';', header=0)

x_mass = ee['title']
y_mass = utils.to_categorical(ee['class'], 2) # eto elsi ne binarnaya
y_mass_bin = ee['class'] # dlya binarnoy

x_mass = pad_sequences(Datasets.string_list_to_sequence(x_mass), NN_init.size_of_array)

x_mass = torch.Tensor(x_mass).to(device)
# y_mass = torch.Tensor(y_mass).to(device)
y_mass = torch.as_tensor(y_mass).to(device)
y_mass_bin = torch.as_tensor(y_mass_bin).to(device)

MyDataset = Datasets.MyDataset(x_mass, y_mass)
MyDatasetBin = Datasets.MyDataset(x_mass, y_mass_bin)

# for ordinary
train_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)

# for binary
train_loader_bin = data.DataLoader(MyDatasetBin, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader_bin = data.DataLoader(MyDatasetBin, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)

# setka
# eto poka ne ispolzuem
# conv_net = torch_test.ConvNet(size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])

# сначала - реснет 34 с одним выходом, потом попробуем обычный, но 101
# resnet50 = Resnet.Resnet("resnet50", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
resnet34 = Resnet.ResnetTest("resnet34", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0], num_classes=1)
# resnet101 = Resnet.Resnet("resnet101", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])

resnet34.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(resnet34.parameters(), lr=NN_init.learning_rate)
conv_net_res = NN_training.training(resnet34, loss_fn, optimizer, train_loader_bin, val_loader_bin, n_epoch=70)

# теперь обычный вариант 101-й
resnet101 = Resnet.Resnet("resnet101", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
resnet101.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet101.parameters(), lr=NN_init.learning_rate)
conv_net_res2 = NN_training.training(resnet101, loss_fn, optimizer, train_loader, val_loader, n_epoch=70)

# LSTM
model_LSTM = LSTM.LSTM_Model(name="LSTM", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0],num_classes=1)
model_LSTM.to(device)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model_LSTM.parameters(), lr=1e-2, amsgrad=True)

model_LSTM_res = NN_training.training_lstm(model_LSTM, loss_fn, optimizer, train_loader_bin, val_loader_bin, n_epoch=150)