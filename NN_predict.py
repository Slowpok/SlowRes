import torch
import NN_training
import Datasets
import NN_init
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


def model_predict(word, name_model, RM=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet101 = torch.load("whole_best_model" + name_model + ".pth", map_location=torch.device(device))
    list1 = []
    ww = Datasets.string_to_seq(word)
    list1.append(ww)
    tensor_word = torch.Tensor(pad_sequences(list1, NN_init.size_of_array))

    tensor_word.to(device)
    if RM:
        hidden = resnet101.init_hidden(device)
        result, hl = resnet101(tensor_word, hidden)

    else:
        result = resnet101(tensor_word)

    result = torch.reshape(result, (-1,))[0].cpu().detach().numpy()
    return 1 if result > 0.5 else 0



