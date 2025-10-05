import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random


# --------------------
# Persistent sets
# --------------------
def persistent_sets(G1, G2, max_nodes=2000):
    """Compute persistent nodes and edges between two graphs with optional sampling."""
    V_star = set(G1.nodes()).intersection(set(G2.nodes()))

    # Sample persistent nodes if too many
    if len(V_star) > max_nodes:
        V_star = set(random.sample(list(V_star), max_nodes))

    E_star_1 = [(u, v) for (u, v) in G1.edges() if u in V_star and v in V_star]
    E_star_2 = [(u, v) for (u, v) in G2.edges() if u in V_star and v in V_star]

    return V_star, E_star_1, E_star_2


def plot_persistent_counts(results, output_dir="plots"):
    """Plot |V*|, |E*| over successive intervals."""
    os.makedirs(output_dir, exist_ok=True)

    intervals = list(range(1, len(results) + 1))
    v_counts = [len(r["V*"]) for r in results]
    e1_counts = [len(r["E*_1"]) for r in results]
    e2_counts = [len(r["E*_2"]) for r in results]

    plt.plot(intervals, v_counts, label="|V*|")
    plt.plot(intervals, e1_counts, label="|E* (Tj)|")
    plt.plot(intervals, e2_counts, label="|E* (Tj+1)|")
    plt.xlabel("Interval Pair (j, j+1)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Persistent Sets Across Intervals")

    filename = os.path.join(output_dir, "persistent_counts.png")
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")


# --------------------
# Similarity matrices
# --------------------
def compute_similarity_matrices(G, V_star, max_pairs=5000):
    """Compute similarity scores for a sample of node pairs in V*."""
    subG = G.subgraph(V_star)

    # Sample pairs to avoid O(n^2)
    nodes = list(V_star)
    pairs = []
    while len(pairs) < max_pairs:
        u, v = random.sample(nodes, 2)
        if u != v:
            pairs.append((u, v))

    results = { "CN": {}, "JC": {}, "AA": {}, "PA": {}, "GD": {} }

    # Compute Common Neighbors + Adamic-Adar
    for u, v in pairs:
        try:
            cn = len(list(nx.common_neighbors(subG, u, v)))
            aa = sum(1 / (nx.degree(subG, z)) for z in nx.common_neighbors(subG, u, v))
        except Exception:
            cn, aa = 0, 0
        results["CN"][(u, v)] = cn
        results["AA"][(u, v)] = aa

    # Jaccard Coefficient
    for u, v, score in nx.jaccard_coefficient(subG, pairs):
        results["JC"][(u, v)] = score

    # Preferential Attachment
    for u, v, score in nx.preferential_attachment(subG, pairs):
        results["PA"][(u, v)] = score

    # Graph Distance (negative shortest path length)
    lengths = dict(nx.all_pairs_shortest_path_length(subG, cutoff=4))  # cutoff to limit cost
    for u, v in pairs:
        if v in lengths.get(u, {}):
            results["GD"][(u, v)] = -lengths[u][v]
        else:
            results["GD"][(u, v)] = 0

    return results


def save_similarity_to_csv(sim_dict, interval_idx, output_dir="plots"):
    """Save similarity scores as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for name, scores in sim_dict.items():
        df = pd.DataFrame(
            [(u, v, val) for (u, v), val in scores.items()],
            columns=["u", "v", name],
        )
        filename = os.path.join(output_dir, f"{name}_interval{interval_idx}.csv")
        df.to_csv(filename, index=False)
        print(f"[Saved] {filename}")
