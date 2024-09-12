import torch
import NN_training
import NN_init
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


def model_predict(word):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet101 = torch.load("whole_best_model.pth")
    tensor_word = torch.Tensor(pad_sequences(NN_training.string_to_seq(word), NN_init.size_of_array)).float()
    tensor_word.to(device)
    resnet101.to(device)
    return resnet101(tensor_word)

