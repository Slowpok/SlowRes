import Datasets
import Resnet
from sklearn.metrics import classification_report
import numpy as np
import NN_init
import pandas as pd
import NN_training
import torch
import torch.utils.data as data
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dannye
ee = pd.read_csv(r"C:\Users\79118\OneDrive\Документы\Нейросет\Материалы датасет3.csv",
                         names=['title', 'class'], encoding='Windows-1251', delimiter=';', header=0)

x_mass = ee['title']
y_mass = utils.to_categorical(ee['class'], 2)

x_mass = pad_sequences(NN_training.string_list_to_sequence(x_mass), NN_init.size_of_array)

x_mass = torch.Tensor(x_mass).to(device)
y_mass = torch.Tensor(y_mass).to(device)

MyDataset = Datasets.MyDataset(x_mass, y_mass)

train_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)
val_loader = data.DataLoader(MyDataset, batch_size=NN_init.batch_size, shuffle=True, pin_memory=False)


# setka
# eto poka ne ispolzuem
# conv_net = torch_test.ConvNet(size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])

# resnet50 = Resnet.Resnet("resnet50", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
resnet34 = Resnet.Resnet("resnet34", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])
# resnet101 = Resnet.Resnet("resnet101", size_token=NN_init.size_of_array, unique_words=x_mass.shape[0])

conv_net = resnet34
conv_net = conv_net.to(device)
loss_fn = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(conv_net.parameters(), lr=NN_init.learning_rate)
conv_net_res = NN_training.training(conv_net, loss_fn, optimizer, train_loader, val_loader, n_epoch=7)

# # Метрики качества
# digit_probabilities = conv_net_res(torch.Tensor(x_mass).long()).detach().numpy()
# print('digit_probabilities.shape:', digit_probabilities.shape)
# # Нашли вероятности принадлежности семплов test-а
#
#
# # Округлили
# predictions = np.argmax(digit_probabilities, axis=1)
# print('predictions.shape:', predictions.shape)
# print('test results:')
# print(classification_report(y_mass.detach().cpu().numpy(), predictions))
