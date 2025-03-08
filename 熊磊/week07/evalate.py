import torch

class Evaluate:
    def __init__(self, Config, model, data):
        self.Config = Config
        self.model = model
        self.data = data

    def eval(self):
        self.model.eval()
        correct, wrong = 0, 0
        for index, batch_data in enumerate(self.data):
            x, y = batch_data
            with torch.no_grad():
                pred = self.model(x)
            
            for y_true, y_pred in zip(y, pred):
                if y_true == torch.argmax(y_pred):
                    correct += 1
                else:
                    wrong += 1
        return correct / (correct + wrong)

