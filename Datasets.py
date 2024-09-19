import torch.utils.data as data
import torch

class MyDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


def string_list_to_sequence(strlist):
    res = []
    for s in strlist:
        # sub_res = []
        # for character in s:
        #     sub_res.append(ord(character))
        res.append(string_to_seq(s))

    return res


def string_to_seq(word):
    sub_res = []
    for character in word:
        sub_res.append(ord(character))

    return sub_res
