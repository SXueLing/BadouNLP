import json
import numpy as np
from loader import load_data
from model import TextClassifyModel, choose_optimizer
from evaluate import evalute
import torch
import random
import datetime
from tqdm import tqdm


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    print("Model type:", config['model_type'])

    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data, test_data = load_data(config)

    model = TextClassifyModel(config)

    if torch.cuda.is_available():
        model.cuda()

    optim = choose_optimizer(config, model)
    if optim is None:
        return
    
    start_timestamp = datetime.datetime.now()
    n_epoches = config['n_epoches']
    for epoch in range(n_epoches):
        model.train()
        train_loss, test_acc = [], 0.0

        for _, batch_data in tqdm(enumerate(train_data), total=len(train_data)):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            x, y = batch_data
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        test_acc = evalute(model, config, test_data)
        print("Epoch: %d, avg loss: %f, test_acc: %f" % (epoch+1, float(np.mean(train_loss)), test_acc))
    end_timestamp = datetime.datetime.now()
    use_time = end_timestamp - start_timestamp
    print("Model type: %s, n_epochs: %d, use time: %d sec." % (config['model_type'], n_epoches, use_time.seconds))
    return


if __name__ == "__main__":
    print("Text Classify ...")
    main(
        config_path='config.json'
    )
