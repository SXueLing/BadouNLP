import numpy as np
def eval(model, test_data):
    model.eval()
    correct = 0
    count = 0
    for x, y in test_data:
        y_pred = model(x)
        count += 1
        if (y_pred > 0.5 and y == 1) or (y_pred<=0.5 and y==0) :
            correct += 1
    print('acc is: {}'.format(correct/count))