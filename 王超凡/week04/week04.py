def load_word_dict(path):
    word_set = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            word = line.split()[0]
            word_set.add(word)
    return word_set

#回溯法
def cut_string(input_str, start_idx, temp_list, result, word_set):
    if start_idx == len(input_str):
        result.append(temp_list.copy())
        return
    for i in range(start_idx, len(input_str)):
        if i == start_idx:
            temp_list.append(input_str[i])
            cut_string(input_str, i + 1, temp_list, result, word_set)
            temp_list.pop()
        else:
            temp_string = input_str[start_idx:i + 1]
            if temp_string in word_set:
                temp_list.append(temp_string)
                cut_string(input_str, i + 1, temp_list, result, word_set)
                temp_list.pop()


def main():
    word_set = load_word_dict("dict.txt")
    input_str = ""
    temp_list = []
    result = []
    input_str = input("input: ")
    while input_str != "exit":
        cut_string(input_str, 0, temp_list, result, word_set)
        print("*******************")
        for split_string in result:
            print(split_string)
        result.clear()
        temp_list.clear()
        input_str = input("input: ")

if __name__ == "__main__":
    main()