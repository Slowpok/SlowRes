from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import codecs

class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        #print(x.shape, " ent")
        x = self.encoder(x).squeeze(2)
        #print(x.shape, " emb")
        out, (ht1, ct1) = self.lstm(x, hidden)
        #print(out.shape, " lstm")
        out = self.dropout(out)
        x = self.fc(out)
        # print(x.shape, " fc")
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))



def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char


def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(16):
        batch_start = np.random.randint(0, len(sequence) - 256)
        chunk = sequence[batch_start: batch_start + 256]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):

    hidden = model.init_hidden(device)
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)
    # print(_.shape, " _ shape")
    # print(hidden.shape, " _ hidden")

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)

        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


TRAIN_TEXT_FILE_PATH = "text_test.txt"
fileObj = codecs.open("text_test.txt", "r", "utf_8_sig" )

text_sample = fileObj.readlines()

text_sample = ' '.join(text_sample)


fileObj.close()

sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    verbose=True,
    factor=0.5
)

n_epochs = 5
loss_avg = []

for epoch in range(n_epochs):
    model.train()
    train, target = get_batch(sequence)
    train = train.permute(1, 0, 2).to(device)
    target = target.permute(1, 0, 2).to(device)
    hidden = model.init_hidden(16)

    output, hidden = model(train, hidden)
    print(output.shape, " _ output")
    loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_avg.append(loss.item())
    if len(loss_avg) >= 5:
        mean_loss = np.mean(loss_avg)
        # print(f'Loss: {mean_loss}')
        scheduler.step(mean_loss)
        loss_avg = []
        model.eval()
        predicted_text = evaluate(model, char_to_idx, idx_to_char)
        # print(predicted_text)


