import random
'''Splitting the training set and validation set'''
def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    num_lines = len(lines)
    num_train = int(0.8 * num_lines)

    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    with open('./data/train_data.txt', 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)

    with open('./data/valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)

split_file('./data/text_classification_dataset.csv')