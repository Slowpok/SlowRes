import NN_init
import torch.nn as nn
import torch


class LSTM_Model(nn.Module):
    def __init__(self, name, unique_words, size_token, num_classes=2):
        super(LSTM_Model, self).__init__()
        self.name = name
        self.lstm_size = 128
        self.embedding_dim = 100
        self.num_layers = 3
        self.num_classes = num_classes

        self.emb1 = nn.Embedding(unique_words, embedding_dim=self.embedding_dim, max_norm=size_token)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_size, num_layers=self.num_layers, dropout=0.1, bias=False)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.lstm_size * size_token, out_features=self.num_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, x, prev_state):
        embed = self.emb1(x.long())
        # print(embed.shape)
        output, state = self.lstm(embed, prev_state)
        #print(output.shape)
        output = self.flat(output)
        #print(output.shape)
        output = self.fc(output)
        #print(output.shape)
        logits = self.sigm(output)
        #print(logits.shape)

        if self.num_classes == 1:
            # out = self.flatten(out)
            logits = torch.reshape(logits, (-1,))
            # out = [1 if x > 0.5 else 0 for x in out]
            # out = torch.as_tensor(out).long()

        return logits, state

    def init_hidden(self):
        return (torch.zeros(self.num_layers, NN_init.size_of_array, self.lstm_size, requires_grad=True),
                torch.zeros(self.num_layers, NN_init.size_of_array, self.lstm_size, requires_grad=True))