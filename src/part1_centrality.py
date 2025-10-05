import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------
# Dataset loading
# --------------------
def load_dataset(path: str, sample_size: int = 10_000):
    """
    Load dataset (source, target, timestamp).
    By default, only loads 'sample_size' rows for faster testing.
    """
    df = pd.read_csv(
        path, sep=" ", names=["source", "target", "timestamp"], nrows=sample_size
    )
    print(f"Loaded {len(df)} edges (sampled)")
    return df


# --------------------
# Time intervals
# --------------------
def split_time_intervals(df, N: int):
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    delta_t = (tmax - tmin) // N
    intervals = [(tmin + i * delta_t, tmin + (i + 1) * delta_t) for i in range(N)]
    intervals[-1] = (intervals[-1][0], tmax)  # last interval inclusive
    return intervals


def build_subgraph(df, interval):
    start, end = interval
    edges = df[(df["timestamp"] >= start) & (df["timestamp"] < end)][
        ["source", "target"]
    ]
    G = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.Graph())
    return G


# --------------------
# Centrality measures
# --------------------
def compute_centralities(G):
    """
    Compute centrality measures with approximations
    for faster execution on large graphs.
    """
    centralities = {
        "degree": nx.degree_centrality(G),
        "closeness": nx.closeness_centrality(G),
    }

    # Approximate betweenness (sample k nodes instead of all)
    if len(G) > 1000:
        centralities["betweenness"] = nx.betweenness_centrality(G, k=500)
    else:
        centralities["betweenness"] = nx.betweenness_centrality(G)

    # Eigenvector (limit iterations)
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality(G, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        centralities["eigenvector"] = {n: 0 for n in G.nodes()}

    # Katz centrality (small alpha for convergence)
    try:
        centralities["katz"] = nx.katz_centrality_numpy(G, alpha=0.005, beta=1.0)
    except Exception:
        centralities["katz"] = {n: 0 for n in G.nodes()}

    return centralities


# --------------------
# Plotting functions
# --------------------
def plot_evolution(intervals, graphs, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    nodes = [len(G.nodes()) for G in graphs]
    edges = [len(G.edges()) for G in graphs]

    plt.plot(range(1, len(intervals) + 1), nodes, label="Nodes")
    plt.plot(range(1, len(intervals) + 1), edges, label="Edges")
    plt.xlabel("Time Interval")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Evolution of |V| and |E| over time")

    filename = os.path.join(output_dir, "evolution.png")
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")


def plot_centrality_histograms(centralities, interval_idx, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    for name, values in centralities.items():
        plt.hist(list(values.values()), bins=30, density=True)
        plt.title(f"Histogram of {name.capitalize()} Centrality (Interval {interval_idx})")
        plt.xlabel("Centrality Value")
        plt.ylabel("Frequency")

        filename = os.path.join(output_dir, f"{name}_interval{interval_idx}.png")
        plt.savefig(filename)
        plt.close()
        print(f"[Saved] {filename}")


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    df = load_dataset("data/sx-stackoverflow.txt", sample_size=10_000)
    N = 5  # number of intervals
    intervals = split_time_intervals(df, N)

    graphs = [build_subgraph(df, interval) for interval in intervals]
    plot_evolution(intervals, graphs)

    for i, G in enumerate(graphs, 1):
        print(f"--- Interval {i} ---")
        centralities = compute_centralities(G)
        plot_centrality_histograms(centralities, i)
