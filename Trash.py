import torch.nn as nn
import torch.nn.functional as F

''' тут и дальше - мусорные классы, на которых было обучение, они просто висят '''
class PytorchTestModel1(nn.Module):
    def __init__(self, size_token, unique_words):
        super().__init__()

        self.em1 = nn.Embedding(unique_words, 32, max_norm=size_token)
        self.conv1d1 = nn.Conv1d(size_token, 100, 3)

    def forward(self, x):

        x = self.em1(x)
        x = F.relu(self.conv1d1(x))

        return x

class ConvNet(nn.Module):
    def __init__(self, unique_words, size_token):
        super(ConvNet, self).__init__()
        #self.layer0 = nn.Embedding(unique_words, NN_init.batch_size, max_norm=size_token)
        self.layer0 = nn.Embedding(unique_words, embedding_dim=100, max_norm=size_token)
        # self.layer1 = nn.Sequential(nn.Conv1d(size_token, 100, kernel_size=5, stride=1, padding=2),
        #                             nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Conv1d(size_token, 32, kernel_size=5, stride=1, padding=2)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool1d(kernel_size=100, stride=2)
        # self.layer2 = nn.Sequential(nn.Conv1d(50, 64, kernel_size=5, stride=1, padding=2),
        #                              nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32, 800)
        self.fc2 = nn.Linear(800, 2)

    def forward(self, x):
        # print("beg emb ", x.size())
        # out = self.layer0(x.long())
        # print(out.size())
        # out = self.layer1(out)
        # print("layer1 ", out.size())
        # out = out.view(out.size(0), -1)
        # print(out.size())
        # out = self.drop_out(out)
        # print(out.size())
        # out = self.fc1(out)
        # print(out.size())
        # out = self.fc2(out)
        #
        # out = F.softmax(out)

        #print("beg emb ", x.size())
        out = self.layer0(x.long())
        #print(out.size())
        out = self.layer2(out)
        #print("layer2 ", out.size())
        out = self.layer3(out)
        #print("layer3 ", out.size())
        out = self.layer4(out)
        #print("layer4 ", out.size())
        out = self.drop_out(out)
        #print("layer5 drop", out.size())
        out = out.view(out.size(0), -1)
        #print("view", out.size())
        out = self.fc1(out)
        #print("layer6 fc1", out.size())
        out = self.fc2(out)
        #print("layer6 fc2", out.size())

        out = F.sigmoid(out)
        # out = torch.flatten(out)
        return out

class PytorchTestModel(nn.Module):
    def __init__(self, size_token, unique_words):
        super().__init__()

        self.em1 = nn.Embedding(unique_words, 32, max_norm=size_token)
        self.conv1d1 = nn.Conv1d(size_token, 100, 3)
        self.conv1d2 = nn.Conv1d(100, 250, 3)

        self.conv1d3 = nn.Conv1d(250, 250, 3, padding="same")
        self.conv1d4 = nn.Conv1d(250, 250, 3, padding="same")
        # tut skladivaem
        self.conv1d5 = nn.Conv1d(250, 250, 3, padding="same")
        self.conv1d6 = nn.Conv1d(250, 250, 3, padding="same")
        # tut skladivaem
        self.conv1d7 = nn.Conv1d(250, 250, 3, padding="same")
        # tut vopros
        # self.avgpool1 = nn.AvgPool1d(5)
        self.lin1 = nn.Linear(50, 256)
        self.drop1 = nn.Dropout(0.25)
        self.lin2 = nn.Linear(256, 1)

    def forward(self, x):

        x = self.em1(x)
        x = F.relu(self.conv1d1(x))
        x = F.relu(self.conv1d2(x))
        out1 = F.max_pool1d(x, 3)

        # x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(output_1)
        # x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(x)
        # output_2 = tf.keras.layers.add([x, output_1])
        x = F.relu(self.conv1d3(out1))
        x = F.relu(self.conv1d4(x))

        # тут я не знаю как складывать
        # out_2 = tf.keras.layers.add([x, out1])

        # x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(output_2)
        # x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(x)
        # output_3 = tf.keras.layers.add([x, output_2])
        x = F.relu(self.conv1d5(x))
        x = F.relu(self.conv1d6(x))
        # тут я снова не знаю как складывать это
        # out_3 = tf.keras.layers.add([x, out_2])

        x = F.relu(self.conv1d7(x))
        x = F.avg_pool1d(x,5)
        # x = self.avgpool1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.drop1(x))
        # в функции потерь уже будет софтмакс, поэтому ничего не делаем
        x = self.lin2(x)

        return x

