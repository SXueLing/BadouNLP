import torch
from loader_eval_data import EvalDataLoad
from seq_vectors import questions_to_vectors
from cosine_calute import cosine_similarity

def model_evaluate(model, train_data):
    vocab = train_data.tokenizer
    questions = train_data.all_questions
    label_map = train_data.label_map
    eval_data = EvalDataLoad("data/valid.txt", vocab)
    wrong, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in eval_data.dataloader:
            input_ids = batch["question"]
            t_label = batch["label"]
            output_vectors = model(input_ids)
            break
    seq_encode = questions_to_vectors(model, train_data)
    p_label, _ = matching_label(output_vectors, seq_encode, questions, label_map)
    for y_t, y_p in zip(t_label, p_label):
        if y_t == y_p:
            correct += 1
        else:
            wrong += 1
    print("本轮评估样本50个，预测正确数：%d,正确率：%f" % (correct, correct / (correct + wrong)))
    return seq_encode


def matching_label(output_vectors, seq_encode, questions, label_map):
    _label = []
    _similarity = []
    for eval_sample in output_vectors:
        index, similarity = cosine_similarity(eval_sample, seq_encode)
        question = questions[index]
        label = label_map[question]
        _label.append(label)
        _similarity.append(similarity)
    return _label, _similarity

