import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def view_by_site_sex(df, vars):
    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [6, 1]}, figsize=(7, 3))

    for k in range(2):
        count_df = df.groupby([vars[k], "Group"]).size().unstack(fill_value=0)
        colors = sns.color_palette("husl", n_colors=len(count_df.columns))
        bottom = np.zeros(len(count_df.index))

        for i, (colname, color) in enumerate(zip(count_df.columns, colors)):
            values = count_df[colname]
            ax[k].bar(count_df.index, values, bottom=bottom, label=colname, color=color)
            for j, v in enumerate(values):
                if v > 0:
                    ax[k].text(
                        j,
                        bottom[j] + v / 2,
                        str(int(v)),
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="white",
                    )
            bottom += values

        ax[k].set_title(f"Count by {vars[k]}", pad=5)
        ax[k].set_xlabel(f"{vars[k]}")
        ax[0].set_ylabel("Count")
        ax[0].set_xticks(np.arange(9))
        ax[0].legend(title="PD Status", ncols=2, loc="upper right")

    plt.tight_layout()
    plt.show()

    return fig


def split_plot(train, test, val, column):
    """
    Seaborn-based visualization of splits
    """
    # Prepare long-format data
    plot_data = pd.concat(
        [
            train[column].to_frame().assign(Split="Train"),
            test[column].to_frame().assign(Split="Test"),
            val[column].to_frame().assign(Split="Validation"),
        ]
    )

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.countplot(data=plot_data, x=column, hue="Split", ax=ax)
    ax.set_title(f"Split Counts by {column}")

    plt.tight_layout()
    plt.show()

    return fig
