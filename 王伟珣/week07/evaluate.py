import torch


def evalute(model, config, test_data):
    model.eval()
    correct, wrong = 0, 0
    for _, batch_data in enumerate(test_data):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        x, y = batch_data
        with torch.no_grad():
            y_pred = model(x)
        
        for y_t, y_p in zip(y, y_pred):
            if y_t == torch.argmax(y_p):
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)
