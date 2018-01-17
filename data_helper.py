import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import pdb
import collections
#过滤函数，
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
#加载数据的函数
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files（使用utf-8存在问题）
    positive = open(positive_data_file, 'rb').read().decode('ISO-8859-1')
    negative = open(negative_data_file, 'rb').read().decode('ISO-8859-1')
    #去掉换行符
    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]
    # 去掉开始或者结尾空格
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_data_labels(data_file, labels_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    data = []
    labels = []
    with open(data_file, 'r', encoding='latin-1') as f:
        data.extend([s.strip() for s in f.readlines()])
        data = [clean_str(s) for s in data]

    with open(labels_file, 'r') as f:
        labels.extend([s.strip() for s in f.readlines()])
        lables = [label.split(',')[1].strip() for label in labels]

    lb = LabelBinarizer()
    y = lb.fit_transform(lables)

    # max_document_length = max([len(x.split(" ")) for x in data])
    # print(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(1000)
    x = np.array(list(vocab_processor.fit_transform(data)))
    return x, y, vocab_processor

#生成batch数据
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

