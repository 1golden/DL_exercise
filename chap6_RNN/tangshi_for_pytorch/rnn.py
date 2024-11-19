import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier initialization
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim)
        )
        self.word_embedding = nn.Embedding(
            vocab_length, embedding_dim
        )  # 一个简单的查找表，用于存储固定词典和大小的嵌入
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embeding_random_intial)
        )

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(
        self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim
    ):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)  # call the weights initial function.

        self.softmax = nn.LogSoftmax()  # the activation function.
        # self.tanh = nn.Tanh()

    def forward(self, sentence, is_test=False):
        sentence = sentence.to(device)
        batch_input = (
            self.word_embedding_lookup(sentence)
            .view(1, -1, self.word_embedding_dim)
            .to(device)
        )
        # print(batch_input.size()) # print the size of the input
        h0 = torch.zeros(2, self.batch_size, self.lstm_dim).to(
            batch_input.device
        )  # 2 是 LSTM 的层数
        c0 = torch.zeros(2, self.batch_size, self.lstm_dim).to(batch_input.device)

        output, (h_n, c_n) = self.rnn_lstm(batch_input)
        # print(output.size())  # print the size of the output.
        out = output.contiguous().view(-1, self.lstm_dim)

        out = F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            output = out
        # print(out)
        return output


def print_layer_shapes_and_parameters(model):
    """
    打印模型每一层的形状及参数

    参数:
        model (nn.Module): 要打印的模型
    """
    for name, layer in model.named_children():
        print(f"Layer Name: {name}")
        for param_name, param in layer.named_parameters():
            print(f"  Parameter Name: {param_name}, Shape: {param.shape}")
        print()


if __name__ == "__main__":
    vocab_len = 10000
    embedding_dim = 300
    lstm_hidden_dim = 128
    batch_sz = 128
    word_embedding = word_embedding(vocab_len, embedding_dim)
    model = RNN_model(
        batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim
    )
    print_layer_shapes_and_parameters(model)
