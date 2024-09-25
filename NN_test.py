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
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from imblearn.over_sampling import SMOTE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dannye
ee = pd.read_csv(filepath_or_buffer=NN_init.finename,
                         names=['title', 'class'], encoding='Windows-1251', delimiter=';', header=0)

x_mass = ee['title']
y_mass = utils.to_categorical(ee['class'], 2) # eto elsi ne binarnaya
y_mass_bin = ee['class'] # dlya binarnoy
# counter for weights
counter = Counter(y_mass_bin)

x_mass = pad_sequences(Datasets.string_list_to_sequence(x_mass), NN_init.size_of_array)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(x_mass, y_mass_bin)

# Convert back to PyTorch tensors
X_smote = torch.tensor(X_smote)
y_smote = torch.tensor(y_smote)

x_mass = torch.Tensor(x_mass).to(device)
X_smote = torch.Tensor(X_smote).to(device)
# y_mass = torch.Tensor(y_mass).to(device)
y_mass = torch.as_tensor(y_mass).to(device)
y_mass_bin = torch.as_tensor(y_mass_bin).to(device)
y_smote = torch.as_tensor(y_smote).to(device)


MyDataset = Datasets.MyDataset(x_mass, y_mass)
MyDatasetBin = Datasets.MyDataset(x_mass, y_mass_bin)
MyDatasetsmote = Datasets.MyDataset(X_smote, y_smote)

# weights1 = [1/counter.get(torch.argmax(y, dim=0).item()) for x, y in MyDataset]
weights = [1/counter.get(y.item()) for x, y in MyDatasetBin]
# sampler1 = WeightedRandomSampler(weights1, num_samples=len(weights1))
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
# sumlen = []
# for x in range(len(weights2)):
#     sumlen.append(1 if weights2[x] != weights1[x] else 0)
# print(sum(sumlen))

# for ordinary
train_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)

# for binary
train_loader_bin = data.DataLoader(MyDatasetBin, batch_size=NN_init.batch_size, pin_memory=False, sampler=sampler)
val_loader_bin = data.DataLoader(MyDatasetBin, batch_size=NN_init.batch_size, pin_memory=False, sampler=sampler)

# for binary not sampler
train_loader_bin_smote = data.DataLoader(MyDatasetsmote, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader_bin_smote = data.DataLoader(MyDatasetsmote, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)

# setka
# eto poka ne ispolzuem
# conv_net = torch_test.ConvNet(size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])

# сначала - реснет 34 с одним выходом, тут используется рандомная выборка из классов
# resnet34 = Resnet.ClassicResnet("resnet34_bin", "resnet34", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0], num_classes=1)
# resnet34.to(device)
# loss_fn = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(resnet34.parameters(), lr=NN_init.learning_rate)
# conv_net_res = NN_training.training(resnet34, loss_fn, optimizer, train_loader_bin, val_loader_bin, n_epoch=70)

# теперь обычный вариант 34-й со взвешенными классами
resnet34_weight = Resnet.Resnet(name="resnet34_weight", nettype="resnet34", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
resnet34_weight.to(device)
class_weights = torch.tensor([0.1, 0.9]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(resnet34_weight.parameters(), lr=NN_init.learning_rate)
conv_net_res2 = NN_training.training(resnet34_weight, loss_fn, optimizer, train_loader, val_loader, n_epoch=70)

resnet50_weight = Resnet.Resnet(name="resnet50_weight", nettype="resnet50", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
resnet50_weight.to(device)
class_weights = torch.tensor([0.1, 0.9]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(resnet50_weight.parameters(), lr=NN_init.learning_rate)
conv_net_res5 = NN_training.training(resnet50_weight, loss_fn, optimizer, train_loader, val_loader, n_epoch=70)

# # теперь вариант 34-й c сгенерированными липовыми данными
# resnet34_smote = Resnet.ResnetTest(name="resnet34_smote", nettype="resnet34", size_token=NN_init.size_of_array, unique_words=X_smote.shape[0], num_classes=1)
# resnet34_smote.to(device)
# loss_fn = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(resnet34_smote.parameters(), lr=NN_init.learning_rate)
# conv_net_res3 = NN_training.training(resnet34_smote, loss_fn, optimizer, train_loader_bin_smote, val_loader_bin_smote, n_epoch=70)

# LSTM
# model_LSTM = LSTM.LSTM_Model(name="LSTM", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0],num_classes=1)
# model_LSTM.to(device)
#
# loss_fn = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(model_LSTM.parameters(), lr=1e-2, amsgrad=True)
#
# model_LSTM_res = NN_training.training_lstm(model_LSTM, loss_fn, optimizer, train_loader_bin, val_loader_bin, n_epoch=150)