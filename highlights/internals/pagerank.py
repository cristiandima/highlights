def pagerank_weighted(graph, initial_value=None, damping=0.85):
    """Calculates PageRank for an undirected weighted graph"""
    CONVERGENCE_THRESHOLD = 0.0001

    if initial_value == None:
        initial_value = 1.0 / len(graph.nodes)

    scores = dict.fromkeys(graph.nodes, initial_value)

    out_degree = {}
    for i in graph.nodes:
        score  = 0
        for k in graph.neighbours[i]:
            score += graph.edges[(i, k)]

        out_degree[i] = 0.000001 if score == 0 else score

    iteration_quantity = 0
    for _ in range(300):
        iteration_quantity += 1
        convergence_achieved = 0
        for i in graph.nodes:
            rank = 1 - damping
            for j in graph.neighbours[i]:
                rank += damping * scores[j] * (graph.edges[(j, i)] / out_degree[j])

            if abs(scores[i] - rank) <= CONVERGENCE_THRESHOLD:
                convergence_achieved += 1

            scores[i] = rank

        if convergence_achieved >= len(graph.nodes):
            break

    return scores