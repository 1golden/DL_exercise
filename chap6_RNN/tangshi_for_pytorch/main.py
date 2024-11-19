import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append(".")
print(sys.path)
import rnn

start_token = "G"
end_token = "E"
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(
        file_name,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(
                    ":"
                )  # 去掉首尾空格，并以冒号分割, 适用于poems.txt文件
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(" ", "")
                if (
                    "_" in content
                    or "(" in content
                    or "（" in content
                    or "《" in content
                    or "[" in content
                    or start_token in content
                    or end_token in content
                ):
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("continue")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频, 返回一个字典，key是词，value是词出现的次数。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 降序排序
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)  # 增加一个空格，作为结束符
    word_int_map = dict(zip(words, range(len(words))))  # 建立词和索引的映射关系
    # print(word_int_map)
    poems_vector = [
        list(map(word_int_map.get, poem)) for poem in poems
    ]  # 对于每首诗句，使用 map() 函数将每个字映射为对应的索引值
    # print(poems_vector[0])
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(
        file_name,
        "r",
        encoding="utf-8",
    ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()  # 去掉首尾空格, 适用于tangshi.txt文件
                if line:
                    content = (
                        line.replace(" " " ", "").replace("，", "").replace("。", "")
                    )
                    if (
                        "_" in content
                        or "(" in content
                        or "（" in content
                        or "《" in content
                        or "[" in content
                        or start_token in content
                        or end_token in content
                    ):
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("continue")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成批次数据

    参数:
        batch_size (int): 批次大小
        poems_vec (list): 诗歌向量列表
        word_to_int (dict): 单词到整数的映射字典

    返回:
        x_batches (list): 输入数据批次列表
        y_batches (list): 目标数据批次列表
    """
    # 计算数据集可以被分成多少个批次
    n_chunk = len(poems_vec) // batch_size
    # 初始化输入数据和目标数据批次列表
    x_batches = []
    y_batches = []
    # 遍历每个批次
    for i in range(n_chunk):
        # 计算当前批次的起始索引和结束索引
        start_index = i * batch_size
        end_index = start_index + batch_size
        # 从诗歌向量列表中获取当前批次的输入数据
        x_data = poems_vec[start_index:end_index]
        # 初始化当前批次的目标数据列表
        y_data = []
        # 遍历当前批次的输入数据中的每一行
        for row in x_data:
            # 将当前行的第二个到最后一个元素添加到目标数据中，并在最后添加当前行的最后一个元素
            y = row[1:]
            y.append(row[-1])
            # 将目标数据添加到当前批次的目标数据列表中
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        # 将当前批次的输入数据添加到输入数据批次列表中
        x_batches.append(x_data)
        # 将当前批次的目标数据添加到目标数据批次列表中
        y_batches.append(y_data)
    # 返回输入数据批次列表和目标数据批次列表
    return x_batches, y_batches


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1(
        "./poems.txt"
    )
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding(
        vocab_length=len(word_to_int) + 1, embedding_dim=100
    ).to(
        device
    )  # 词嵌入层
    rnn_model = rnn.RNN_model(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    ).to(device)

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()  # 负对数似然损失函数

    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(
            BATCH_SIZE, poems_vector, word_to_int
        )
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]  # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1))).to(device)
                y = Variable(torch.from_numpy(y)).to(device)
                pre = rnn_model(x)
                loss += loss_fun(pre, y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print(
                        "prediction", pre.data.tolist()
                    )  # the following  three line can print the output and the prediction
                    print(
                        "b_y       ", y.data.tolist()
                    )  # And you need to take a screenshot and then past is to your homework paper.
                    print("*" * 30)
            loss = loss / BATCH_SIZE
            print(
                "epoch  ", epoch, "batch number", batch, "loss is: ", loss.data.tolist()
            )
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1) #已弃用的函数
            torch.nn.utils.clip_grad_value_(rnn_model.parameters(), clip_value=1.0)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), "./poem_generator_rnn")
                print("finish  save model")


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige = []
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split("。")
    for s in poem_sentences:
        if s != "" and len(s) > 10:
            print(s + "。")


def gen_poem(begin_word):
    """
    打印模型每一层的形状及参数

    参数:
        model (nn.Module): 要打印的模型

    返回:
        None
    """
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1("./poems.txt")
    word_embedding = rnn.word_embedding(
        vocab_length=len(word_int_map) + 1, embedding_dim=100
    ).to(device)
    rnn_model = rnn.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    ).to(device)

    rnn_model.load_state_dict(torch.load("./poem_generator_rnn"))

    # 指定开始的字

    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = Variable(torch.from_numpy(input)).to(device)
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem


# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。

# 输出每一首可能的诗歌
pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))
pretty_print_poem(gen_poem("月"))
