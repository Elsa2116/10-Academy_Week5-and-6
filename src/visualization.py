from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_class_distribution_plot(dist: pd.DataFrame, target: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist[target].astype(str), dist["count"])
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, row in dist.reset_index(drop=True).iterrows():
        ax.text(i, row["count"], f"{row['percentage']:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_top_country_fraud_plot(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    country_rates = (
        df.groupby("country")["class"]
        .agg(fraud_rate="mean", transactions="count")
        .query("transactions >= 20")
        .sort_values("fraud_rate", ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(country_rates["country"], country_rates["fraud_rate"] * 100)
    ax.invert_yaxis()
    ax.set_title("Top countries by fraud rate")
    ax.set_xlabel("Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_numeric_by_target_boxplot(df: pd.DataFrame, value: str, target: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column=value, by=target, ax=ax)
    ax.set_title(f"{value} by target")
    fig.suptitle("")
    ax.set_xlabel(target)
    ax.set_ylabel(value)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
