import torch
import NN_training
import NN_init
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


def model_predict(word):
    resnet101 = torch.load("whole_best_model.pth")
    tensor_word = torch.Tensor(pad_sequences(NN_training.string_to_seq(word), NN_init.size_of_array)).float()

    return resnet101(tensor_word)

