import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import chain
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import islice


def load_log_from_csv(file_path):
    df = pd.read_csv(file_path)
    log = []
    for case_id, events in df.groupby('case_id')['event_name']:
        log.append(events.tolist())
    return log


file_path = 'test_logs/FixPermitLog.csv'
log = load_log_from_csv(file_path)

for i, trace in enumerate(log[:10]):
    print(f"Trace {i + 1}: {trace}")


def build_ubcg(log):
    G = nx.DiGraph()
    edges = set()
    for trace in log:
        for i in range(len(trace) - 1):
            event = trace[i]
            for successor in trace[i + 1:]:
                if (event, successor) not in edges:
                    G.add_edge(event, successor)
                    edges.add((event, successor))
    return G


def extract_choices(log):
    events = list(set(chain.from_iterable(log)))
    choices = []
    for trace in log:
        choices.append([event in trace for event in events])
    return np.array(choices), events


def find_mec(log, choices, events):
    corr_matrix = np.corrcoef(choices, rowvar=False)
    G = nx.Graph()
    valid_pairs = set()

    for trace in log:
        for i in range(len(trace)):
            for j in range(i + 1, len(trace)):
                valid_pairs.add((trace[i], trace[j]))

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if not np.isnan(corr_matrix[i, j]) and not np.isinf(corr_matrix[i, j]):
                if abs(corr_matrix[i, j]) > 0.5:
                    if (events[i], events[j]) in valid_pairs or (events[j], events[i]) in valid_pairs:
                        G.add_edge(events[i], events[j], weight=corr_matrix[i, j])
    return G


def combine_mec_ubcg(mec, ubcg):
    combined = ubcg.copy()
    for src, tgt, data in mec.edges(data=True):
        if combined.has_edge(src, tgt) or combined.has_edge(tgt, src):
            if combined.has_edge(src, tgt):
                edge = combined[src][tgt]
            else:
                edge = combined[tgt][src]

            if 'weight' in edge:
                edge['weight'] += data['weight']
            else:
                edge['weight'] = data['weight']
    return combined


def estimate_causal_effects(log, choices, events, dependencies):
    causal_effects = {}
    valid_pairs = set()

    for trace in log:
        for i in range(len(trace)):
            for j in range(i + 1, len(trace)):
                valid_pairs.add((trace[i], trace[j]))

    for src in events:
        for tgt in events:
            if src != tgt and (src, tgt) in valid_pairs:
                if (src, tgt) in dependencies or (tgt, src) in dependencies:
                    X = choices[:, events.index(src)].reshape(-1, 1)
                    y = choices[:, events.index(tgt)]
                    if len(set(X.flatten())) > 1 and len(set(y)) > 1:
                        model = LinearRegression().fit(X, y)
                        causal_effects[(src, tgt)] = model.coef_[0]
    return causal_effects


def plot_graph_with_weights(G, title="Graph with Weights"):
    pos = nx.spring_layout(G, k=4, iterations=50)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}

    plt.figure(figsize=(12, 8))

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.margins(0.2)
    plt.show()


ubcg = build_ubcg(log)
choices, events = extract_choices(log)
mec = find_mec(log, choices, events)
combined_graph = combine_mec_ubcg(mec, ubcg)
causal_effects = estimate_causal_effects(log, choices, events, mec.edges)


plot_graph_with_weights(ubcg, "Upper Bound Causal Graph (UBCG)")
plot_graph_with_weights(mec, "Markov Equivalence Class (MEC)")
plot_graph_with_weights(combined_graph, "Combined Causal Graph")

print("Causal Effects:")
for edge, effect in causal_effects.items():
    print(f"Effect of '{edge[0]}' on '{edge[1]}': {effect:.2f}")


data = pd.DataFrame.from_dict(causal_effects, orient='index', columns=['Effect'])
data.index = pd.MultiIndex.from_tuples(data.index)


pivot_table = data['Effect'].unstack()


pivot_table = pivot_table.reindex(events, axis=0).reindex(events, axis=1)


plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar_kws={'label': 'Causal Effect'},
                      annot_kws={"size": 10})

plt.xticks(rotation=45, ha='right')

plt.title('Causal Effects Heatmap')
plt.show()


range_edges = dict(islice(causal_effects.items(), 10))


G = nx.DiGraph()

for (src, tgt), weight in range_edges.items():
    G.add_edge(src, tgt, weight=weight)


pos = nx.spring_layout(G, k=4, iterations=50)


node_colors = ['skyblue' if weight < 0 else 'lightcoral' for weight in nx.get_edge_attributes(G, 'weight').values()]
node_sizes = [3000 for _ in G.nodes()]


plt.figure(figsize=(14, 10))
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color='lightgrey', font_size=10, font_weight='bold', arrowsize=20, edge_color=node_colors)


edge_labels = {(src, tgt): f"{data['weight']:.2f}" for src, tgt, data in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('Causal Effects Graph')
plt.margins(0.2)
plt.show()
