"""Exploratory reports and plots for preprocessed URL data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from url_phishing_classification.config import (
    REDIRECT_KEYWORDS,
    SHORTENING_SERVICES,
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_TLDS,
)

LENGTH_FEATURES = {
    "URL": "url_length",
    "Host": "hostname_length",
    "Domain": "domain_length",
    "Subdomain": "subdomain_length",
    "Path": "path_length",
    "Query": "query_length",
    "Fragment": "fragment_length",
}

PRESENCE_FEATURES = {
    "Path": "path",
    "Query": "query",
    "Fragment": "fragment",
    "Params": "params",
    "Subdomain": "subdomain",
}

COUNT_FEATURES = {
    "Path segments": "path_segment_count",
    "Query parameters": "query_parameters",
}

COMPOSITION_FEATURES = {
    "URL digit ratio": "digit_ratio",
    "URL letter ratio": "url_letter_ratio",
    "URL non-alphanumeric ratio": "url_non_alphanumeric_ratio",
    "Host digit ratio": "host_digit_ratio",
    "Host letter ratio": "host_letter_ratio",
    "Path digit ratio": "path_digit_ratio",
    "Path letter ratio": "path_letter_ratio",
    "Query digit ratio": "query_digit_ratio",
    "Query letter ratio": "query_letter_ratio",
}

SEPARATOR_AND_HIERARCHY_FEATURES = {
    "Dots": "dot_count",
    "Hyphens": "hyphen_count",
    "Underscores": "underscore_count",
    "Slashes": "slash_count",
    "Equals signs": "equals_count",
    "Ampersands": "ampersand_count",
    "Non-alphanumeric characters": "non_alphanumeric_count",
    "Host labels": "host_label_count",
    "Subdomain labels": "subdomain_labels",
}

ENTROPY_COMPONENTS = {
    "URL": ("url_entropy", "url_norm_entropy"),
    "Hostname": ("hostname_entropy", "hostname_norm_entropy"),
    "Domain": ("domain_entropy", "domain_norm_entropy"),
    "Subdomain": ("subdomain_entropy", "subdomain_norm_entropy"),
    "Path": ("path_entropy", "path_norm_entropy"),
    "Query": ("query_entropy", "query_norm_entropy"),
    "Fragment": ("fragment_entropy", "fragment_norm_entropy"),
}

STRUCTURAL_INDICATORS = {
    "Contains @": "has_at",
    "Percent encoding": "has_percent_encoding",
    "Punycode host": "has_punycode",
    "IP-literal host": "is_ip_literal_host",
    "No host": "no_host",
    "Unusual port": "unusual_port",
    "Malformed port": "has_malformed_port",
    "Unicode": "has_unicode",
    "Specified port": "has_port",
    "Repeated separator": "has_repeated_separator",
    "Double slash in path": "double_slash_in_path",
}

PARSED_TEXT_COMPONENTS = {
    "Query": ("query", "has_query"),
    "Fragment": ("fragment", "has_fragment"),
    "Params": ("params", "has_params"),
}

TAIL_FEATURES = {
    "URL length": "url_length",
    "Query length": "query_length",
    "URL entropy": "url_entropy",
    "Digit ratio": "digit_ratio",
    "Dots": "dot_count",
    "Hyphens": "hyphen_count",
    "Slashes": "slash_count",
    "Equals signs": "equals_count",
    "Ampersands": "ampersand_count",
}

CORRELATION_FAMILIES = {
    "Lengths, separators, and query structure": [
        "url_length",
        "hostname_length",
        "domain_length",
        "subdomain_length",
        "path_length",
        "query_length",
        "fragment_length",
        "dot_count",
        "hyphen_count",
        "underscore_count",
        "slash_count",
        "equals_count",
        "ampersand_count",
        "non_alphanumeric_count",
        "query_parameters",
    ],
    "Raw and normalized entropy": [
        "url_entropy",
        "url_norm_entropy",
        "hostname_entropy",
        "hostname_norm_entropy",
        "domain_entropy",
        "domain_norm_entropy",
        "subdomain_entropy",
        "subdomain_norm_entropy",
        "path_entropy",
        "path_norm_entropy",
        "query_entropy",
        "query_norm_entropy",
        "fragment_entropy",
        "fragment_norm_entropy",
    ],
    "Subdomain and host hierarchy": [
        "subdomain_length",
        "host_label_count",
        "dot_count",
        "subdomain_labels",
    ],
    "Presence and paired counts": [
        "has_query",
        "query_length",
        "query_parameters",
        "has_fragment",
        "fragment_length",
        "has_params",
    ],
    "Character-composition ratios": [
        "digit_ratio",
        "host_digit_ratio",
        "host_letter_ratio",
        "path_digit_ratio",
        "path_letter_ratio",
        "query_digit_ratio",
        "query_letter_ratio",
    ],
    "Rare structural flags": [
        "has_at",
        "has_percent_encoding",
        "has_punycode",
        "is_ip_literal_host",
        "has_port",
        "has_repeated_separator",
        "double_slash_in_path",
    ],
}

RISK_CURVE_FEATURES = {
    "URL length": "url_length",
    "Host length": "hostname_length",
    "Path length": "path_length",
    "Query length": "query_length",
    "URL digit ratio": "digit_ratio",
    "URL entropy": "url_entropy",
    "Path segments": "path_segment_count",
    "Host digit ratio": "host_digit_ratio",
    "Query digit ratio": "query_digit_ratio",
}


def deduplication_stats(
    df: pd.DataFrame,
    original_col: str = "url",
    normalized_col: str = "normalized_url",
    label_col: str = "label",
) -> pd.DataFrame:
    """Compare duplicate URLs and label conflicts before and after normalization."""
    results = []

    for name, url_col in (
        ("original", original_col),
        ("normalized", normalized_col),
    ):
        rows = df.loc[df[url_col].notna(), [url_col, label_col]]
        groups = rows.groupby(url_col, sort=False)[label_col]

        group_sizes = groups.transform("size")
        label_counts = groups.transform(lambda labels: labels.nunique(dropna=False))
        conflicting = group_sizes.gt(1) & label_counts.gt(1)

        results.append(
            {
                "url_type": name,
                "duplicate_rows": int(rows[url_col].duplicated().sum()),
                "rows_in_conflicting_groups": int(conflicting.sum()),
                "conflicting_groups": int(rows.loc[conflicting, url_col].nunique()),
            }
        )

    return pd.DataFrame(results).set_index("url_type")


def host_uses_service(host: str, service: str) -> bool:
    """Check whether a hostname matches a shortening service or its subdomain."""
    return host == service or host.endswith(f".{service}")


def label_1_pct(df: pd.DataFrame, matches: pd.Series) -> float:
    """Return the percentage of label-1 URLs among matching URLs."""
    if not matches.any():
        return 0.0

    return df.loc[matches, "label"].eq(1).mean() * 100


def plot_length_and_component_eda(df, label_col="label", bins=30):
    """Plot URL-component lengths, presence rates, and structural counts by label."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    for ax, (name, feature) in zip(axes.flat, LENGTH_FEATURES.items()):
        values = df[feature].dropna()
        display_max = max(1, values.quantile(0.99))

        for label, color in zip(labels, colors):
            group = df.loc[df[label_col] == label, feature].dropna().clip(upper=display_max)
            ax.hist(group, bins=bins, range=(0, display_max), alpha=0.5, color=color, label=str(label))

        ax.set_title(name)
        ax.set_xlabel("Length")

    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, [f"label = {label}" for label in legend_labels], loc="upper center", ncol=len(labels))
    fig.suptitle("Component lengths by label (display clipped at each feature's 99th percentile)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    first_fig = fig

    presence = pd.DataFrame({name: df[column].fillna("").ne("") for name, column in PRESENCE_FEATURES.items()})
    presence_by_label = presence.groupby(df[label_col]).mean().T

    fig, (presence_ax, count_ax) = plt.subplots(1, 2, figsize=(14, 4.5))
    presence_by_label.plot(kind="bar", ax=presence_ax, color=colors[: len(labels)], width=0.8)
    presence_ax.set_title("Component presence rate")
    presence_ax.set_xlabel("")
    presence_ax.set_ylabel("Share of URLs")
    presence_ax.yaxis.set_major_formatter(PercentFormatter(1))
    presence_ax.legend(title="label")
    presence_ax.tick_params(axis="x", rotation=0)

    count_data = [
        df.loc[df[label_col] == label, feature].dropna() for feature in COUNT_FEATURES.values() for label in labels
    ]
    positions = [
        count_index * (len(labels) + 1) + label_index + 1
        for count_index in range(len(COUNT_FEATURES))
        for label_index in range(len(labels))
    ]
    box = count_ax.boxplot(count_data, positions=positions, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors * len(COUNT_FEATURES)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    centers = [count_index * (len(labels) + 1) + (len(labels) + 1) / 2 for count_index in range(len(COUNT_FEATURES))]
    count_ax.set_xticks(centers, COUNT_FEATURES.keys())
    count_ax.set_title("Structural counts by label")
    count_ax.set_ylabel("Count")
    count_ax.legend(
        [Rectangle((0, 0), 1, 1, color=color, alpha=0.55) for color in colors[: len(labels)]],
        [f"label = {label}" for label in labels],
        title="label",
    )
    fig.tight_layout()
    return first_fig, fig


def plot_character_and_structure_eda(df, label_col="label"):
    """Compare composition, structure, and paired entropy features by label."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    label_colors = dict(zip(labels, colors))

    fig, (ratio_ax, count_ax) = plt.subplots(1, 2, figsize=(18, 5))

    ratio_by_label = df.groupby(label_col)[list(COMPOSITION_FEATURES.values())].mean().T
    ratio_by_label.index = COMPOSITION_FEATURES.keys()
    ratio_by_label.plot(kind="bar", ax=ratio_ax, color=colors[: len(labels)], width=0.8)
    ratio_ax.set_title("Mean character-composition ratios")
    ratio_ax.set_xlabel("")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.set_ylim(0, 1)
    ratio_ax.tick_params(axis="x", rotation=35)
    ratio_ax.legend(title="label")

    count_data = [
        df.loc[df[label_col] == label, feature].dropna()
        for feature in SEPARATOR_AND_HIERARCHY_FEATURES.values()
        for label in labels
    ]
    positions = [
        feature_index * (len(labels) + 1) + label_index + 1
        for feature_index in range(len(SEPARATOR_AND_HIERARCHY_FEATURES))
        for label_index in range(len(labels))
    ]
    box = count_ax.boxplot(count_data, positions=positions, patch_artist=True, showfliers=False)
    box_colors = [label_colors[label] for _ in SEPARATOR_AND_HIERARCHY_FEATURES for label in labels]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    centers = [
        feature_index * (len(labels) + 1) + (len(labels) + 1) / 2
        for feature_index in range(len(SEPARATOR_AND_HIERARCHY_FEATURES))
    ]
    count_ax.set_xticks(centers, SEPARATOR_AND_HIERARCHY_FEATURES.keys(), rotation=35, ha="right")
    count_ax.set_title("Separator and hierarchy counts by label")
    count_ax.set_ylabel("Count (outliers hidden)")
    count_ax.legend(
        [Rectangle((0, 0), 1, 1, color=label_colors[label], alpha=0.55) for label in labels],
        [f"label = {label}" for label in labels],
        title="label",
    )
    fig.tight_layout()
    first_fig = fig

    n_cols = 3
    n_rows = (len(ENTROPY_COMPONENTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    for ax, (component, (entropy_feature, normalized_feature)) in zip(axes.flat, ENTROPY_COMPONENTS.items()):
        for label, color in zip(labels, colors):
            values = df.loc[df[label_col] == label, [entropy_feature, normalized_feature]].dropna()
            ax.scatter(
                values[entropy_feature],
                values[normalized_feature],
                alpha=0.1,
                s=12,
                color=color,
                label=f"label = {label}",
            )

        correlation = df[[entropy_feature, normalized_feature]].corr().iloc[0, 1]
        ax.set_title(f"{component}: entropy vs normalized entropy (r={correlation:.2f})")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Normalized entropy")

    for ax in axes.flat[len(ENTROPY_COMPONENTS) :]:
        ax.axis("off")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(labels))
    fig.suptitle("Raw and normalized entropy are paired candidate signals", y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return first_fig, fig


def plot_entropy_distributions_by_label(df, label_col="label", bins=30):
    """Plot overlaid label distributions for raw and normalized entropy features."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    entropy_features = [(component, feature, "Entropy") for component, (feature, _) in ENTROPY_COMPONENTS.items()] + [
        (component, feature, "Normalized entropy") for component, (_, feature) in ENTROPY_COMPONENTS.items()
    ]

    n_cols = 4
    n_rows = (len(entropy_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)

    for ax, (component, feature, entropy_type) in zip(axes.flat, entropy_features):
        for label, color in zip(labels, colors):
            values = df.loc[df[label_col] == label, feature].dropna()
            ax.hist(values, bins=bins, density=True, alpha=0.5, color=color, label=f"label = {label}")

        ax.set_title(f"{component} {entropy_type.lower()}")
        ax.set_xlabel(entropy_type)
        ax.set_ylabel("Density")

    for ax in axes.flat[len(entropy_features) :]:
        ax.axis("off")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
    fig.suptitle("Entropy distributions by label", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def summarize_numeric_features_by_label(df, label_col="label", features=None):
    """Return class-conditioned numeric summaries, including missing and zero rates."""
    if features is None:
        features = [
            feature for feature in df.columns if feature != label_col and pd.api.types.is_numeric_dtype(df[feature])
        ]

    records = []
    for feature in features:
        for label in sorted(df[label_col].dropna().unique()):
            values = pd.to_numeric(df.loc[df[label_col] == label, feature], errors="coerce").astype(float)
            observed = values.dropna()
            q25, q75 = observed.quantile([0.25, 0.75]) if not observed.empty else (float("nan"), float("nan"))
            records.append(
                {
                    "feature": feature,
                    "label": label,
                    "median": observed.median(),
                    "mean": observed.mean(),
                    "iqr": q75 - q25,
                    "p90": observed.quantile(0.90),
                    "p95": observed.quantile(0.95),
                    "missing_rate": values.isna().mean(),
                    "zero_rate": values.eq(0).sum() / len(values),
                }
            )

    return pd.DataFrame(records).set_index(["feature", "label"]).round(3)


def plot_binary_indicators_by_label(df, indicators=STRUCTURAL_INDICATORS, label_col="label"):
    """Plot binary indicator prevalence with both percentage and class count."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    rates = df.groupby(label_col)[list(indicators.values())].mean().T
    counts = df.groupby(label_col)[list(indicators.values())].sum().T
    rates.index = indicators.keys()
    counts.index = indicators.keys()

    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.8 / len(labels)
    x_positions = list(range(len(indicators)))
    for label_index, label in enumerate(labels):
        offset = (label_index - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            [position + offset for position in x_positions],
            rates[label],
            width=width,
            color=colors[label_index],
            label=f"label = {label}",
        )
        for bar, count in zip(bars, counts[label]):
            ax.annotate(
                f"{bar.get_height():.1%}\n(n={count:,.0f})",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions, indicators.keys(), rotation=30, ha="right")
    ax.set_ylim(0, min(1, rates.to_numpy().max() * 1.25 + 0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of URLs")
    ax.set_title("Structural indicator prevalence by label")
    ax.legend(title="label")
    fig.tight_layout()
    return fig


def plot_port_distribution_by_label(df, label_col="label"):
    """Compare specified ports with ECDFs and robust, outlier-hidden boxplots."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    port_data = [df.loc[(df[label_col] == label) & df["port"].gt(0), "port"].dropna() for label in labels]

    fig, (ecdf_ax, box_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color, values in zip(labels, colors, port_data):
        values = values.sort_values()
        if not values.empty:
            ecdf_ax.step(
                values,
                [(index + 1) / len(values) for index in range(len(values))],
                where="post",
                color=color,
                label=f"label = {label}",
            )

    ecdf_ax.set_xscale("log")
    ecdf_ax.set_xlabel("Specified port (log scale)")
    ecdf_ax.set_ylabel("Cumulative share")
    ecdf_ax.set_title("ECDF of specified ports")
    ecdf_ax.legend()

    box = box_ax.boxplot(
        port_data, tick_labels=[f"label = {label}" for label in labels], patch_artist=True, showfliers=False
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    box_ax.set_ylabel("Specified port")
    box_ax.set_title("Specified ports: robust central comparison")
    fig.tight_layout()
    return fig


def plot_scheme_and_suffix_patterns(df, label_col="label", suffix_top_n=10):
    """Describe low-cardinality schemes and the most common public suffixes by label."""
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    def plot_category_rates(ax, feature, title, top_n=None):
        categories = df[feature].fillna("(missing)")
        if top_n is not None:
            top_categories = categories.value_counts().head(top_n).index
            categories = categories.where(categories.isin(top_categories), "Other")

        counts = pd.crosstab(categories, df[label_col]).reindex(columns=labels, fill_value=0)
        rates = counts.div(counts.sum(axis=0), axis=1)
        x_positions = list(range(len(counts)))
        width = 0.8 / len(labels)
        for label_index, label in enumerate(labels):
            offset = (label_index - (len(labels) - 1) / 2) * width
            bars = ax.bar(
                [position + offset for position in x_positions],
                rates[label],
                width=width,
                color=colors[label_index],
                label=f"label = {label}",
            )
            for bar, count in zip(bars, counts[label]):
                ax.annotate(
                    f"{bar.get_height():.1%}\n(n={count:,.0f})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(x_positions, counts.index, rotation=35, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Share within label")
        ax.set_title(title)

    fig, (scheme_ax, suffix_ax) = plt.subplots(1, 2, figsize=(17, 5))
    plot_category_rates(scheme_ax, "scheme", "Scheme pattern by label")
    plot_category_rates(suffix_ax, "suffix", f"Top {suffix_top_n} suffix patterns by label", top_n=suffix_top_n)
    scheme_ax.legend(title="label")
    fig.tight_layout()
    return fig


def summarize_parsed_text_sparsity(df, label_col="label"):
    """Summarize empty-string, missing, and present rates for sparse parsed text."""
    records = []
    for component, (text_feature, presence_feature) in PARSED_TEXT_COMPONENTS.items():
        for label in sorted(df[label_col].dropna().unique()):
            values = df.loc[df[label_col] == label, text_feature]
            records.append(
                {
                    "component": component,
                    "label": label,
                    "rows": len(values),
                    "empty_string_rate": values.fillna("").eq("").mean(),
                    "missing_rate": values.isna().mean(),
                    "presence_rate": df.loc[values.index, presence_feature].mean(),
                }
            )

    return pd.DataFrame(records).set_index(["component", "label"]).round(3)


def plot_tail_ecdfs_by_label(df, label_col="label"):
    """Compare the tails of a small set of continuous features without binning."""
    plot_features = {
        "URL length": "url_length",
        "Query length": "query_length",
        "URL entropy": "url_entropy",
        "Digit ratio": "digit_ratio",
    }
    labels = sorted(df[label_col].dropna().unique())
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (name, feature) in zip(axes.flat, plot_features.items()):
        for label, color in zip(labels, colors):
            values = df.loc[df[label_col] == label, feature].dropna().sort_values()
            ax.step(
                values,
                [(index + 1) / len(values) for index in range(len(values))],
                where="post",
                color=color,
                label=f"label = {label}",
            )

        if feature == "url_length":
            ax.set_xscale("log")
        elif feature == "query_length":
            ax.set_xscale("symlog", linthresh=1)

        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Cumulative share")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Tail review by label", y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def review_extreme_urls(df, features=TAIL_FEATURES, label_col="label", top_n=5):
    """Return top records per class and feature for parser/outlier auditing, not modelling."""
    audit_columns = [
        label_col,
        "url",
        "scheme",
        "hostname",
        "query",
        "fragment",
        "params",
        "no_host",
        "has_malformed_port",
    ]
    records = []
    for name, feature in features.items():
        for label in sorted(df[label_col].dropna().unique()):
            extremes = df.loc[df[label_col] == label].nlargest(top_n, feature)[audit_columns + [feature]].copy()
            extremes.insert(0, "review_feature", name)
            extremes.rename(columns={feature: "feature_value"}, inplace=True)
            records.append(extremes)

    return pd.concat(records, ignore_index=True).sort_values(
        ["review_feature", label_col, "feature_value"], ascending=[True, True, False]
    )


def high_spearman_pairs(df, features, threshold=0.75):
    """Return non-duplicate feature pairs with strong absolute Spearman correlation."""
    correlation = df[features].astype(float).corr(method="spearman")
    pairs = []
    for row_index, left in enumerate(correlation.columns):
        for right in correlation.columns[row_index + 1 :]:
            value = correlation.loc[left, right]
            if abs(value) >= threshold:
                pairs.append({"feature_1": left, "feature_2": right, "spearman_rho": value})

    return (
        pd.DataFrame(pairs, columns=["feature_1", "feature_2", "spearman_rho"])
        .sort_values("spearman_rho", key=lambda values: values.abs(), ascending=False)
        .round(3)
    )


def plot_spearman_families(df, families=CORRELATION_FAMILIES):
    """Plot Spearman matrices for related feature families, with a shared colour scale."""
    n_cols = 2
    n_rows = (len(families) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), squeeze=False, layout="constrained")
    image = None
    for ax, (name, features) in zip(axes.flat, families.items()):
        correlation = df[features].astype(float).corr(method="spearman")
        image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(features)), features, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(features)), features, fontsize=8)
        ax.set_title(name)

    for ax in axes.flat[len(families) :]:
        ax.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Spearman correlation")
    fig.suptitle("Feature-family redundancy review", y=0.98)
    return fig


def plot_quantile_risk_curves(
    df, features=RISK_CURVE_FEATURES, label_col="label", n_bins=10, min_bin_size=50
) -> tuple[Figure, pd.DataFrame]:
    """Plot label-1 rate by quantile bin, annotating each bin's sample count."""
    positive_label = sorted(df[label_col].dropna().unique())[-1]
    records = []
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), squeeze=False)

    for ax, (name, feature) in zip(axes.flat, features.items()):
        values = df[[feature, label_col]].dropna().copy()
        bin_count = min(n_bins, values[feature].nunique())
        if bin_count < 2:
            ax.set_visible(False)
            continue

        values["bin"] = pd.qcut(values[feature], q=bin_count, duplicates="drop")
        summary = (
            values.groupby("bin", observed=True)
            .agg(phishing_rate=(label_col, lambda labels: labels.eq(positive_label).mean()), count=(label_col, "size"))
            .reset_index()
        )
        summary["feature"] = feature
        records.append(summary)

        x = range(len(summary))
        reliable = summary["count"].ge(min_bin_size)
        ax.plot(x, summary["phishing_rate"], color="crimson", linewidth=1.5)
        ax.scatter(
            [index for index, keep in enumerate(reliable) if keep],
            summary.loc[reliable, "phishing_rate"],
            color="crimson",
            s=28,
        )
        ax.scatter(
            [index for index, keep in enumerate(reliable) if not keep],
            summary.loc[~reliable, "phishing_rate"],
            facecolors="none",
            edgecolors="crimson",
            s=34,
        )
        for index, row in summary.iterrows():
            ax.annotate(
                f"n={row['count']:,}",
                (index, row["phishing_rate"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

        ax.set_xticks(list(x), [str(interval) for interval in summary["bin"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, min(1, summary["phishing_rate"].max() * 1.2 + 0.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(name)
        ax.set_xlabel("Feature quantile range")
        ax.set_ylabel("Phishing rate")

    for ax in axes.flat[len(features) :]:
        ax.axis("off")
    fig.suptitle("Phishing rate by feature range (hollow markers: fewer than 50 URLs)", y=1.01)
    fig.tight_layout()
    return fig, pd.concat(records, ignore_index=True)


def signal_prevalence_reports(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize engineered risk signals and their matching terms."""
    shortening_matches = pd.DataFrame(
        {
            service: df["hostname"].map(
                lambda host, service=service: host_uses_service(host, service) if pd.notna(host) else False
            )
            for service in SHORTENING_SERVICES
        }
    )
    redirect_matches = pd.DataFrame(
        {keyword: df["url"].str.contains(keyword, regex=False) for keyword in REDIRECT_KEYWORDS}
    )
    suspicious_keyword_matches = pd.DataFrame(
        {keyword: df["url"].str.contains(keyword, regex=False) for keyword in SUSPICIOUS_KEYWORDS}
    )
    suspicious_tld_matches = pd.DataFrame({tld: df["suffix"].eq(tld) for tld in SUSPICIOUS_TLDS})

    signal_matches = {
        feature: df[feature]
        for feature in (
            "has_shortening_service",
            "has_redirect_keyword",
            "has_suspicious_keyword",
            "has_suspicious_tld",
        )
    }
    signal_report = pd.DataFrame(
        {
            "feature": signal_matches.keys(),
            "matched_urls": [matches.sum() for matches in signal_matches.values()],
            "prevalence_pct": [matches.mean() * 100 for matches in signal_matches.values()],
            "label_1_pct": [label_1_pct(df, matches) for matches in signal_matches.values()],
        }
    ).round({"prevalence_pct": 2, "label_1_pct": 2})

    term_report = pd.concat(
        [
            pd.DataFrame(
                {
                    "term": matches.columns,
                    "matched_urls": matches.sum().values,
                    "label_1_pct": [label_1_pct(df, matches[term]) for term in matches.columns],
                    "signal": signal,
                }
            )
            for signal, matches in {
                "shortening_service": shortening_matches,
                "redirect_keyword": redirect_matches,
                "suspicious_keyword": suspicious_keyword_matches,
                "suspicious_tld": suspicious_tld_matches,
            }.items()
        ],
        ignore_index=True,
    ).assign(
        prevalence_pct=lambda report: (report["matched_urls"] / len(df) * 100).round(2),
        label_1_pct=lambda report: report["label_1_pct"].round(2),
    )
    return signal_report, term_report


def summarize_high_spearman_pairs(
    df: pd.DataFrame, families=CORRELATION_FAMILIES, threshold: float = 0.75
) -> pd.DataFrame:
    """Find strong Spearman pairs across the configured feature families."""
    features = list(dict.fromkeys(feature for family in families.values() for feature in family))
    return high_spearman_pairs(df, features, threshold=threshold)
