import sys, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_loss(fpath, title=None):
    df = pd.read_csv(fpath)
    if title is None:
        title = os.path.basename(fpath)
    ax = sns.lineplot(x='epoch', y='train_loss', data=df, label="Training")
    ax = sns.lineplot(x='epoch', y='test_loss', data=df, label="Testing")
    ax.set(xlabel='Epochs', ylabel='Loss', title=title)
    plt.show()



if __name__ == '__main__':
    path = sys.argv[1]
    plot_loss(path)
