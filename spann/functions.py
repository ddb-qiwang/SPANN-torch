import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import f, norm, ttest_ind


def ftest(s1, s2):
    """
    f-test for two distribtuions
    """
    print("Null Hypothesis:var(s1)=var(s2)，α=0.05")
    F = np.var(s1) / np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2 * abs(0.5 - f.cdf(F, v1, v2))
    print(p_val)
    if p_val < 0.05:
        print("Reject the Null Hypothesis.")
        equal_var = False
    else:
        print("Accept the Null Hypothesis.")
        equal_var = True
    return equal_var


def ttest_ind_fun(s1, s2):
    """
    t-test for two distributions
    """
    equal_var = ftest(s1, s2)
    print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
    ttest, pval = ttest_ind(s1, s2, equal_var=equal_var)
    if pval < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval


def distribution_plot(Escore_bank, threshold_test, bins=20, ylim=180):
    """
    show the distribution of spatial data E-scores

    :Escore_bank: learned E-scores for spatial data
    :threshold_test: the test threshold learned by SPANN
    :bins: how many slices the E-score distribution is divided into, default=20
    :ylim: the maximum value of y axis, dafault=180
    """
    sns.set_style("white")
    fig = plt.figure(figsize=(6, 6))
    # sns.set(font_scale=2)
    # sns.distplot(ensemble, kde=False, bins=15)  # , fit=stats.norm
    sns.distplot(Escore_bank, kde=False, color="green", label="Known", bins=bins)  # , fit=stats.norm
    plt.axvline(threshold_test, color="brown", label=r"$\delta$ = %.2f" % threshold_test, linewidth=2)  # )
    plt.xlim(0, 1.0)
    plt.ylim(0, ylim)

    # plt.xlabel("Entropy", fontdict={'fontweight': 'bold', 'fontsize': 20}, x=0.5, y=1.2)

    plt.ylabel("Number of cells", fontdict={"fontsize": 20}, x=-0.02, y=0.5)
    # plt.title("Neonatal rib", fontdict={'fontweight': 'bold', 'fontsize': 15}, loc="left")
    plt.title("E-score", fontdict={"fontsize": 20}, x=0.5, y=1.1)
    ax = plt.gca()
    ax.spines["right"].set_color("none")  # 取消右坐标轴
    ax.spines["top"].set_color("none")  # 取消上坐标轴
    ax.xaxis.set_ticks_position("bottom")  # 设底坐标轴为x轴
    ax.yaxis.set_ticks_position("left")  # 设左坐标轴为y轴
    ax.spines["bottom"].set_position(("data", -0))  # 将底坐标轴放在纵轴为0的地方
    # ax.spines['left'].set_position(('data', -0))  # 将左坐标轴放在横轴为0的地方
    plt.yticks(fontproperties="Arial", size=18)  # 设置大小及加粗
    plt.xticks(fontproperties="Arial", size=18)
    plt.legend()
    plt.show()

    return fig
