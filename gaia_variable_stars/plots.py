import matplotlib.pyplot as plt
import seaborn as sns

from gaia_variable_stars.config import CLASSES, FEATURES, FIGURES_ROOT

sns.set_theme(style="whitegrid")


def save(name):
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.gcf().savefig(FIGURES_ROOT / f"{name}.png", dpi=150, bbox_inches="tight")


def counts(stars):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=stars, x="class_name", order=CLASSES)
    plt.xlabel("Class")
    plt.ylabel("Stars")
    save("counts")


def points(stars):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=stars, x="class_name", y="points", order=CLASSES)
    plt.xlabel("Class")
    plt.ylabel("G-band epochs")
    save("points")


def cmd(stars):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=stars, x="color", y="mag", hue="class_name", hue_order=CLASSES, alpha=0.7
    )
    plt.gca().invert_yaxis()
    plt.xlabel("BP-RP color")
    plt.ylabel("Mean G magnitude")
    save("cmd")


def curves(data):
    """Raw light curve of the best-sampled star in each class."""
    size = data.groupby(["class_name", "source_id"]).size().rename("n").reset_index()
    top = (
        size.sort_values("n", ascending=False)
        .drop_duplicates("class_name")
        .set_index("class_name")["source_id"]
    )

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, name in zip(axes.ravel(), CLASSES):
        star = data[data["source_id"] == top[name]]
        ax.scatter(star["g_transit_time"], star["g_transit_mag"], s=12)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("G magnitude")

    save("curves")


def fold(data, feat):

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, name in zip(axes.ravel(), CLASSES):
        group = feat[feat["class_name"] == name].sort_values("power")
        row = group.iloc[len(group) // 2]
        star = data[data["source_id"] == row["source_id"]]
        phase = (star["g_transit_time"] % row["period"]) / row["period"]

        ax.scatter(phase, star["g_transit_mag"], s=12)
        ax.invert_yaxis()
        ax.set_title(f"{name}: P = {row['period']:.3f} d")
        ax.set_xlabel("Phase")
        ax.set_ylabel("G magnitude")

    save("fold")


def dist(feat):
    cols = 5
    rows = -(-len(FEATURES) // cols)
    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = axes.ravel()

    for ax, name in zip(axes, FEATURES):
        sns.boxplot(
            data=feat, x="class_name", y=name, order=CLASSES, showfliers=False, ax=ax
        )
        ax.set_xlabel("")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=40)

    for ax in axes[len(FEATURES):]:
        ax.set_visible(False)

    save("dist")


def corr(feat):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        feat[FEATURES].corr(method="spearman"),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    save("corr")


def confusion(matrix):
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save("confusion")


def importance(rank):
    rank = rank.sort_values("drop")

    plt.figure(figsize=(8, 5))
    plt.barh(rank["feature"], rank["drop"], xerr=rank["std"], capsize=3)
    plt.xlabel("Drop in test macro F1 when shuffled")
    save("importance")


def wrong(data, feat, test, n=4):
    """Phase-folded curves of test stars the model got wrong."""
    bad = test[test["true"] != test["pred"]].head(n)

    if bad.empty:
        return

    _, axes = plt.subplots(1, len(bad), figsize=(4 * len(bad), 3.5), squeeze=False)

    for ax, (_, row) in zip(axes[0], bad.iterrows()):
        period = feat.loc[feat["source_id"] == row["source_id"], "period"].iloc[0]
        star = data[data["source_id"] == row["source_id"]]
        phase = (star["g_transit_time"] % period) / period

        ax.scatter(phase, star["g_transit_mag"], s=12)
        ax.invert_yaxis()
        ax.set_title(f"true: {row['true']}\npred: {row['pred']}", fontsize=9)
        ax.set_xlabel("Phase")

    save("wrong")

    _, axes = plt.subplots(1, len(bad), figsize=(4 * len(bad), 3.5), squeeze=False)

    for ax, (_, row) in zip(axes[0], bad.iterrows()):
        period = feat.loc[feat["source_id"] == row["source_id"], "period"].iloc[0]
        star = data[data["source_id"] == row["source_id"]]
        phase = (star["g_transit_time"] % period) / period

        ax.scatter(phase, star["g_transit_mag"], s=12)
        ax.invert_yaxis()
        ax.set_title(f"true: {row['true']}\npred: {row['pred']}", fontsize=9)
        ax.set_xlabel("Phase")

    save("wrong")