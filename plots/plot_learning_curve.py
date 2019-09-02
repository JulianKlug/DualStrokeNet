import sys, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curve(fpath, title=None):
    df = pd.read_csv(fpath)
    if title is None:
        title = os.path.basename(fpath)
    ax = sns.lineplot(x='epoch', y='train_dice', data=df, label="Training")
    ax = sns.lineplot(x='epoch', y='test_dice', data=df, label="Testing")
    ax.set(xlabel='Epochs', ylabel='Dice', title=title)
    plt.show()



if __name__ == '__main__':
    path = sys.argv[1]
    plot_learning_curve(path)
