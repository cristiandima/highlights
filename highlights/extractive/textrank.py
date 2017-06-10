import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from highlights.internals.graph import Graph
from highlights.internals.pagerank import pagerank_weighted
from highlights.internals.helpers import summary_length, NLP

def _textrank_scores(sentences):
    tf_idf = TfidfVectorizer(stop_words='english').fit_transform(sentences)
    sim_graph = cosine_similarity(tf_idf, dense_output=True)

    graph = Graph()
    for i in range(len(sentences)):
        graph.add_node(i)

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                graph.add_edge((i,j), sim_graph[i, j])

    scores = pagerank_weighted(graph)
    return scores


def textrank(text, len_func=summary_length):
    sentences = [sent.text for sent in NLP(text).sents]
    scores = _textrank_scores(sentences)

    sum_len = len_func(len(sentences))
    sent_scores = [(scores[i], s) for i, s in enumerate(sentences)]
    top_sentences = sorted(sent_scores, reverse=True)[:sum_len]

    return [s[1] for s in top_sentences]
