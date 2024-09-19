import Datasets
import Resnet
import NN_init
import pandas as pd
import NN_training
import torch
import torch.utils.data as data
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dannye
ee = pd.read_csv(r"C:\Users\79118\OneDrive\Документы\Нейросет\Материалы датасет3.csv",
                         names=['title', 'class'], encoding='Windows-1251', delimiter=';', header=0)

x_mass = ee['title']
y_mass = ee['class'] # dlya binarnoy

x_mass = pad_sequences(Datasets.string_list_to_sequence(x_mass), NN_init.size_of_array)

x_mass = torch.Tensor(x_mass).to(device)
y_mass = torch.as_tensor(y_mass).to(device)

MyDataset = Datasets.MyDataset(x_mass, y_mass)

train_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_LSTM = LSTM.LSTM_Model(name="LSTM", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0],num_classes=1)
model_LSTM.to(device)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model_LSTM.parameters(), lr=1e-2, amsgrad=True)

model_LSTM_res = NN_training.training_lstm(model_LSTM, loss_fn, optimizer, train_loader, val_loader, n_epoch=7)
