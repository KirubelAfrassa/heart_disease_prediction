import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


def plot_histograms(df, train_data):
    analysis_cols = [col for col in df.columns if col != "output"]

    # creating for a plot to put subplots in.
    fig, axes = plt.subplots(nrows=len(analysis_cols), ncols=2, figsize=(12, 60))
    for index, col in enumerate(analysis_cols):
        sns.histplot(train_data[train_data["output"] == 0][col], color="green", alpha=0.7, ax=axes[index, 0])
        sns.histplot(train_data[train_data["output"] == 1][col], color="red", alpha=0.7, ax=axes[index, 1])
        axes[index, 0].set_xlabel(str(col))
        axes[index, 0].set_ylabel('Count')
        if index == 0:
            axes[index, 0].set_title('Less chance of heart attack')
            axes[index, 1].set_title('More chance of heart attack')
        axes[index, 1].set_xlabel(str(col))
        axes[index, 1].set_ylabel('Count')
        plt.savefig("data_presentation/output.png")
