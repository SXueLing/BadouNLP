import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def pd_split_cvs_to_json(fileName,
                         dst_dir,
                         train_file='train_data.json',
                         test_file='test_data.json',
                         test_size=0.1):
    """
    该方法会将cvs文件分割为json文件，为模式训练使用
    in_args:
    file_name: cvs文件名称
    train_file: 训练数据的jso文件名
    test_file: 测试数据的json文件名
    test_size: 调用sklearn中的train_test_split方法，测试数据集的比例。如果0.1为总数据量的10%
    """
    train_data_path = os.path.join(dst_dir, train_file)
    test_data_path = os.path.join(dst_dir, test_file)

    df = pd.read_csv(fileName)
    label_counts = df['label'].value_counts().to_dict()
    print(label_counts)
    columns = df.columns.values
    df['label'] = df['label'].replace({1: "好评", 0: "差评"})
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=7)
    # with open(train_file, 'w', encoding='utf-8') as f:
    #     for _, row in train_df.iterrows():
    #         json_line = json.dumps({"label": row['label'], "review": row['review']}, ensure_ascii=False)
    #         f.write(json_line + '\n')

    # with open(test_file, 'w', encoding='utf-8') as f:
    #     for _, row in test_df.iterrows():
    #         json_line = json.dumps({"label": row['label'], "review": row['review']}, ensure_ascii=False)
    #         f.write(json_line + '\n')
    with open(dst_dir + train_file, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            data_dict = {}
            for column in columns:
                data_dict[column] = row[column]
            json_line = json.dumps(data_dict, ensure_ascii=False)
            f.write(json_line + '\n')
    with open(dst_dir + test_file, 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            data_dict = {}
            for column in columns:
                data_dict[column] = row[column]
            json_line = json.dumps(data_dict, ensure_ascii=False)
            f.write(json_line + '\n')

    print("data split done")
    return train_data_path, test_data_path

def main():
    # df = load_data('nn_pipline/data/classify_data.csv')
    # split_data(df)
    pd_split_cvs_to_json('data/classify_data.csv', 'data/')

if __name__ == '__main__':
    main()
