import pandas as pd
import itertools as ittls
import networkx as nx
import matplotlib.pyplot as plt


def load_log_from_csv(file_path):
    df = pd.read_csv(file_path)
    log = []
    for case_id, events in df.groupby('case_id')['event_name']:
        log.append(events.tolist())
    return log


file_path = 'test_logs/log.csv'
log = load_log_from_csv(file_path)

for i, trace in enumerate(log[:10]):
    print("{} trace: {}".format(i+1, trace))


def return_unique_activities(input_log):
    uniq_act = set()
    for trace in input_log:
        for act in trace:
            uniq_act.add(act)
    return list(uniq_act)


def return_all_direct_succession(input_log):
    successions = set()
    for trace in input_log:
        for i in range(len(trace) - 1):
            successions.add((trace[i], trace[i + 1]))
    return list(successions)


def get_causality_matrix(input_log):
    uniq_activity = return_unique_activities(input_log)
    dir_successions = return_all_direct_succession(input_log)

    causality_matrix = {}
    for key1 in uniq_activity:
        causality_matrix[key1] = {}
    for a1 in uniq_activity:
        for a2 in uniq_activity:
            if (a1, a2) in dir_successions and (a2, a1) in dir_successions:
                causality_matrix[a1][a2] = "||"
            elif (a1, a2) in dir_successions and (a2, a1) not in dir_successions:
                causality_matrix[a1][a2] = "->"
            elif (a1, a2) not in dir_successions and (a2, a1) in dir_successions:
                causality_matrix[a1][a2] = "<-"
            else:
                causality_matrix[a1][a2] = "#"
    return pd.DataFrame(causality_matrix).transpose().sort_index().sort_index(axis=1)


def return_transitions(input_log):
    T_L = return_unique_activities(input_log)
    T_I = list(set([trace[0] for trace in input_log]))
    T_O = list(set([trace[-1] for trace in input_log]))
    return (T_L, T_I, T_O)


def find_subsets(lst):
    subsets = []
    for i in range(1, len(lst) + 1):
        subsets.extend(list(ittls.combinations(lst, i)))
    return subsets


def return_X_L(input_log):
    TR_ALL, TR_INPUT, TR_OUT = return_transitions(input_log)
    c_matrix = get_causality_matrix(input_log)
    A_s = find_subsets(TR_ALL)[:-1]
    AB_s = []
    for A in A_s:
        for B in find_subsets(list(set(TR_ALL) - set(A))):
            AB_s.append((A, B))
    def check_direction(A, B):
        for a in A:
            for b in B:
                if c_matrix.loc()[a][b] != "->":
                    return False
        return True

    def check_no_relation(A, B):
        for a in A:
            for b in B:
                if c_matrix.loc()[a][b] != "#":
                    return False
        return True

    filtered_AB_s = []
    for A, B in AB_s:
        if check_direction(A, B) and check_no_relation(A, A) and check_no_relation(B, B):
            filtered_AB_s.append((A, B))
    X_L = list(filtered_AB_s)
    return X_L


def return_Y_L(input_log):
    X_L = return_X_L(input_log)
    Y_L = X_L.copy()
    remove_lst =[]
    for i in range(0, len(X_L)-1):
        for j in range(i+1, len(X_L)):
            if set(X_L[i][0]).issubset(set(X_L[j][0])):
                if set(X_L[i][1]).issubset(set(X_L[j][1])):
                    if X_L[i] not in remove_lst:
                        remove_lst.append(X_L[i])
    for rem_elem in remove_lst:
        Y_L.remove(rem_elem)
    return Y_L


def return_P_L(input_log):
    T_L, T_I, T_O = return_transitions(input_log)
    Y_L = return_Y_L(input_log)
    P_L = []
    for i in range(len(Y_L)):
        P_L.append(("P" + str(i + 1), {"From": Y_L[i][0], "To": Y_L[i][1]}))
    P_L.insert(0, ("start", {"From": (), "To": [elem for elem in T_I]}))
    P_L.append(("end", {"From": [elem for elem in T_O], "To": ()}))
    return P_L


def return_F_L(input_log):
    P_L = return_P_L(input_log)
    F_L = [(a, p_name) for p_name, p_attr in P_L for a in p_attr["From"]]
    F_L += [(p_name, a) for p_name, p_attr in P_L for a in p_attr["To"]]
    return F_L


def alpha_miner(input_log):
    T_L, T_I, T_O = return_transitions(input_log)
    P_L = return_P_L(input_log)
    F_L = return_F_L(input_log)
    return (P_L, T_L, F_L)


P_L, T_L, F_L = alpha_miner(log)
print("Places:")
print(P_L)
print("Transitions:")
print(T_L)
print("Flows:")
print(F_L)


G = nx.DiGraph()
places = P_L
transitions = T_L
flows = F_L

for place, _ in places:
    G.add_node(place, type='place')

for transition in transitions:
    G.add_node(transition, type='transition')

for source, target in flows:
    G.add_edge(source, target)

place_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'place']
transition_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'transition']

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, node_color='lightblue', node_shape='o', node_size=500)
nx.draw_networkx_nodes(G, pos, nodelist=transition_nodes, node_color='lightgreen', node_shape='s', node_size=500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

plt.axis('off')
plt.show()