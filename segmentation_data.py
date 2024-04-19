# -*- coding: utf-8 -*-
import random

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def shuffle_content(content):
    random.shuffle(content)
    return content

def split_content(content):
    total_length = len(content)
    train_size = int(0.8 * total_length)
    valid_size = int(0.1 * total_length)

    train_data = content[:train_size]
    valid_data = content[train_size:train_size + valid_size]
    test_data = content[train_size + valid_size:]

    return train_data, valid_data, test_data

def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.writelines(data)

# 读取文件内容
content = read_file('data.json')

# 打乱顺序
content = shuffle_content(content)

# 切分成三部分
train_data, dev_data, test_data = split_content(content)

# 写入不同的文件
write_to_file('data/train.json', train_data)
write_to_file('data/dev.json', dev_data)
write_to_file('data/test.json', test_data)