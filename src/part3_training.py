import pandas as pd
import numpy as np
import os


# --------------------
# Accuracy calculation
# --------------------
def compute_accuracy(sim_scores, E_train, E_test, RX):
    """
    Compute accuracy for a similarity measure.
    sim_scores: dict {(u,v): score}
    E_train: training edges (set of tuples)
    E_test: testing edges (set of tuples)
    RX: range of accepted similarity scores (list of (low, high))
    """
    # Predicted edges based on RX
    E_pred = set()
    for (u, v), score in sim_scores.items():
        for low, high in RX:
            if low <= score <= high:
                E_pred.add((u, v))

    # Ground truth = testing edges
    E = set(E_test)
    E0_size = len(sim_scores)  # number of pairs we evaluated
    if E0_size == 0:
        return 0.0

    # Confusion matrix components
    TP = len(E_pred & E)
    FN = len(E - E_pred)
    FP = len(E_pred - E)
    TN = E0_size - TP - FP - FN

    # TPR & TNR
    TPR = TP / len(E) if len(E) > 0 else 0
    TNR = TN / (E0_size - len(E)) if (E0_size - len(E)) > 0 else 0

    lam = len(E) / E0_size
    ACC = lam * TPR + (1 - lam) * TNR
    return ACC


# --------------------
# Training algorithm
# --------------------
def train_optimal_ranges(sim_scores, E_train, E_test, bins=10):
    """
    Train to find the best similarity score range RX.
    We discretize similarity scores into bins and search for best interval.
    """
    values = list(sim_scores.values())
    if not values:
        return [], 0.0

    min_val, max_val = min(values), max(values)
    thresholds = np.linspace(min_val, max_val, bins + 1)

    best_acc = 0.0
    best_range = []

    # Try all possible single intervals
    for i in range(len(thresholds) - 1):
        RX = [(thresholds[i], thresholds[i + 1])]
        acc = compute_accuracy(sim_scores, E_train, E_test, RX)
        if acc > best_acc:
            best_acc = acc
            best_range = RX

    # Convert to clean floats for output
    best_range_clean = [(float(low), float(high)) for (low, high) in best_range]
    return best_range_clean, float(best_acc)


# --------------------
# Save training results
# --------------------
def save_training_results(results, output_dir="plots"):
    """
    Save results as a clean CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    filename = os.path.join(output_dir, "training_results.csv")
    df.to_csv(filename, index=False)
    print(f"[Saved] {filename}")


# --------------------
# Ranking plot 
# --------------------
def plot_average_accuracy(results, output_dir="plots"):
    """
    Plot average accuracy per similarity measure across all intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)

    avg_acc = df.groupby("Measure")["TrainAccuracy"].mean().sort_values(ascending=False)

    import matplotlib.pyplot as plt
    avg_acc.plot(kind="bar", figsize=(8, 5), title="Average Training Accuracy per Measure")
    plt.ylabel("Accuracy")
    plt.xlabel("Similarity Measure")

    filename = os.path.join(output_dir, "accuracy_ranking.png")
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")
