import matplotlib.pyplot as plt
import seaborn as sns


def score_barplot(df, score='mape', save_path=None, y_lims=None):
    plt.figure(figsize=(4, 3))

    df = df.reset_index()

    plt.xticks(rotation=0)
    ax = sns.barplot(x="target", y=score, hue='regressor', data=df, palette='Set1')

    # add value labels on top of the bars
    for p in ax.containers:
        fmt = '%.4f'
        ax.bar_label(p, label_type='edge', fontsize=11, padding=2, fmt=fmt, rotation=90)

    try:
        plt.ylim(y_lims[0], y_lims[1])
    except:
        pass

    plt.xlabel('')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{score}.pdf')
    plt.show()