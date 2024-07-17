import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib._color_data as mcd


folder = "checkpoint/CVQMAE"
input_dir = os.path.join(folder, "Distributions_3DPW")
output_dir = os.path.join(folder, "analyze_distributions")

file_list = glob.glob(os.path.join(input_dir, "*.csv"))

metric_list = ["v2v", "mpjpe", "pampjpe"]


df_all_scores = pd.DataFrame(columns=["method", "v2v", "mpjpe", "pampjpe"])

for file in file_list:

    method = Path(file).stem

    df_subject = pd.read_csv(file)

    for index, row in df_subject.iterrows():

        [_, v2v, mpjpe, pampjpe] = row

        df_all_scores.loc[len(df_all_scores)] = [method, v2v, mpjpe, pampjpe]

df_all_scores.to_csv(f"{output_dir}/raw_results.csv")

method_list = list(df_all_scores["method"].unique())
method_list.sort()

df_summary = pd.DataFrame(
    columns=[
        "method",
        "metric",
        "mean",
        "std",
        "CI_95_mean",
        "median",
        "sample_size",
    ]
)

for method in method_list:

    for metric in metric_list:

        test = df_all_scores["method"] == method

        df = df_all_scores[test]

        if len(df) != 0:

            mean = np.mean(df[metric])
            median = np.median(df[metric])
            std = np.std(df[metric], ddof=1)
            CI = 1.96 * std / len(df)
            sample_size = len(df)

            row = [method, metric, mean, std, CI, median, sample_size]

            df_summary.loc[len(df_summary)] = row

            print(method + " " + metric + ": ", end="")
            print("%.2f" % mean)

df_summary.to_csv(f"{output_dir}/results_summary.csv")

methods_ranked = {}

for j, metric in enumerate(metric_list):

    test = df_summary["metric"] == metric

    df_ranked = df_summary[test][["method", "mean"]]
    df_ranked = df_ranked.sort_values(by=["mean"], ascending=False)

    methods_ranked[metric] = list(df_ranked["method"])


colors = [
    "tab:grey",
    "tab:green",
    "tab:blue",
    "tab:red",
    "tab:olive",
    "tab:cyan",
    "tab:orange",
    "tab:purple",
    "xkcd:orchid",
    "xkcd:aqua",
    "xkcd:indigo",
    "xkcd:salmon",
]

my_pal = {}
for i, method in enumerate(method_list):
    my_pal[method] = colors[i]

# legend
custom_lines = []
for key in my_pal.keys():
    custom_lines.append(Line2D([0], [0], color=my_pal[key], lw=4))


plt.close("all")

bp_width = 0.2
bp_space_method = 0.15
bp_space = 1

color = ["white", "white"]
# color = ['pink', 'lightblue']

# fig, axs = plt.subplots(2,2)
# inds = [(1,1), (1,0), (0,1), (0,0)]
cpt = 0

for metric in metric_list:

    # ax = axs[inds[cpt]]
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    pos = 0
    cpt2 = 0

    cond_list_plot = []

    means = []

    for n, method in enumerate(methods_ranked[metric]):

        test = df_all_scores["method"] == method

        df = df_all_scores[test]

        res = np.array(df[metric])

        bp = ax.boxplot(
            res,
            positions=[pos],
            notch=True,
            widths=bp_width,
            patch_artist=True,
            boxprops=dict(facecolor=color[0]),
            showfliers=False,
        )

        for median in bp["medians"]:
            median.set_color("black")

        means.append(np.mean(res))
        ax.scatter(pos, np.mean(res), marker=".", color="black", s=15, zorder=3)

        ax.annotate("{:.2f}".format(np.mean(res)), (pos - 0.09, 5.1), fontsize=20)

        parts = ax.violinplot(
            res, positions=[pos], showmeans=False, showmedians=False, showextrema=False
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(my_pal[method])
            pc.set_edgecolor("black")
            pc.set_alpha(1)

        cpt2 += 1

        pos += bp_space

        cond_list_plot.append(method)

    ax.plot(means, ":k", alpha=0.5)

    ax.set_title(metric, fontsize=15)
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.75)
    ax.yaxis.grid(True, linestyle="--", which="minor", color="lightgrey", alpha=0.75)

    ax.set_xticklabels(cond_list_plot, rotation=0, fontsize=15)

    plt.yticks(fontsize=15)

    if metric == "v2v":
        plt.ylim([20, 160])
    elif metric == "mpjpe":
        plt.ylim([20, 130])
    else:
        plt.ylim([10, 85])

    cpt += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}.svg")
    plt.savefig(f"{output_dir}/{metric}.pdf")
