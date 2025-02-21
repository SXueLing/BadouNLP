from torch.nn.functional import cosine_similarity

def compute_similarity(model, text1, text2):
    model.eval()
    with torch.no_grad():
        embed1 = model([text1])
        embed2 = model([text2])
        similarity = cosine_similarity(embed1, embed2)
    return similarity.item()