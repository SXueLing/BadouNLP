import torch

def questions_to_vectors(model, train_data):
    question = train_data.all_questions
    vectors = []
    for q in question:
        vectors.append(train_data.encode(q))
    input_ides = torch.tensor(vectors, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        output = model(input_ides)
    return list(output)
