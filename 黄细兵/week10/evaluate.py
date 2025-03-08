import torch

from loader import load_vocab

class Evaluator():
    def __init__(self, config, model):
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.model = model

    def eval(self, sentence, tokenizer):
        reverse_vocab = dict((y, x) for x, y in self.vocab.items())
        self.model.eval()
        with torch.no_grad():
            target = ""
            while target != "\n" and len(sentence) < 30:
                sentence += target
                # x = [self.vocab.get(char, self.vocab["<UNK>"]) for char in sentence]
                x = tokenizer.encode(sentence, add_special_tokens=False)

                x = torch.LongTensor([x])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = self.model(x)[0][-1]
                maxIndex = int(torch.argmax(y))
                # target = reverse_vocab[maxIndex]
                target = ''.join(tokenizer.decode(maxIndex))
        return sentence
