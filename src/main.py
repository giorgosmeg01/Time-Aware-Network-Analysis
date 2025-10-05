from part1_centrality import load_dataset, split_time_intervals, build_subgraph
from part2_link_prediction import persistent_sets, compute_similarity_matrices
from part3_training import train_optimal_ranges, save_training_results, plot_average_accuracy

if __name__ == "__main__":
    # --- Load sample data ---
    df = load_dataset("data/sx-stackoverflow.txt", sample_size=10_000)
    N = 5
    intervals = split_time_intervals(df, N)
    graphs = [build_subgraph(df, interval) for interval in intervals]

    # --- Part III ---
    training_results = []
    for j in range(len(graphs) - 1):
        V_star, E_star_1, E_star_2 = persistent_sets(graphs[j], graphs[j+1])
        sim = compute_similarity_matrices(graphs[j], V_star)

        for name, scores in sim.items():
            RX, acc_train = train_optimal_ranges(scores, E_star_1, E_star_2, bins=10)
            training_results.append({
                "Interval": f"{j}-{j+1}",
                "Measure": name,
                "BestRange": RX,
                "TrainAccuracy": acc_train
            })

    save_training_results(training_results)
    plot_average_accuracy(training_results)
