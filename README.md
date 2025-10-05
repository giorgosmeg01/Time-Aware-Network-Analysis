# Time-Aware Network Centrality Measures and Link Prediction in Temporal Networks
This repository contains the implementation for the Social Network Analysis assignment on Time-Aware Network Centrality Measures & Link Prediction, assigned by Dr. Dionisios N. Sotiropoulos on April 8, 2022. The project analyzes the Stack Overflow Temporal Network, a directed graph with timestamped edges representing interactions (e.g., answers, comments) between users.
The analysis partitions the network's timeline into non-overlapping periods, computes subgraphs for each period, evaluates centrality measures, and performs link prediction using similarity metrics. It includes graphical representations of network evolution and accuracy evaluations for link prediction models.
Key objectives:

Partition the temporal network and visualize its evolution.
Compute and plot centrality measures (Degree, Closeness, Betweenness, Eigenvector, Katz).
Implement similarity-based link prediction (Graph Distance, Common Neighbors, Jaccard's Coefficient, Adamic/Adar, Preferential Attachment).
Train and evaluate classifiers for optimal prediction accuracy on training and testing sets.

# Features
# Part I: Temporal Network Partitioning and Centrality Analysis 

Partition the time interval [t_min, t_max] into N non-overlapping periods.
Represent subgraphs G[t_{j-1}, t_j] as undirected graphs.
Visualize the evolution of node and edge counts over time.
Compute and plot histograms for centrality measures in each subgraph.

# Part II: Persistent Nodes and Similarity Metrics 

Identify persistent nodes and edges across successive periods.
Visualize volumes of persistent sets.
Compute similarity matrices for pairs of persistent nodes: SGD (Graph Distance), SCN (Common Neighbors), SJC (Jaccard's Coefficient), SA (Adamic/Adar), SPA (Preferential Attachment).

# Part III: Link Prediction and Evaluation 

Train an algorithm to find optimal range sets R*_X maximizing accuracy on training edges.
Evaluate and rank training/testing accuracies for each similarity measure.
Use metrics like TPR, TNR, and weighted accuracy (ACC).

# Dataset

sx-stackoverflow.txt: Timestamped edges from the Stack Overflow network (source ID, target ID, timestamp).
Download from SNAP.

# Technologies Used

Python 3.8+: Core implementation.
NetworkX: For graph construction, centrality calculations, and similarity metrics.
Matplotlib/Seaborn: For visualizations (histograms, time-series plots).
Pandas/NumPy: Data handling and timestamp partitioning.
SciPy: For optimization in training algorithm (e.g., maximizing ACC).
