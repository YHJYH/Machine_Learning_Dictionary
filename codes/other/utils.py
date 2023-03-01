from typing import List
from matplotlib import pyplot as plt

def  draw_loss_graph(train_loss: List[float], test_loss: List[float], loss_type: str, model_dataset: str):
    epoch = [i for i in range(1, len(train_loss)+1)]
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, test_loss, label='test_loss')
    best_y =  min(test_loss)
    best_x = test_loss.index(best_y)
    plt.annotate('early stopping', xy=(best_x, best_y), xytext=(best_x, best_y+(max(test_loss)-best_y)/2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
    plt.axis()
    plt.legend()
    plt.title(model_dataset)
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.show()
