"""
Broadly speaking, this is an implementation of the algorithm presented in
TextRank: Bringing Order into Texts - Mihalcea (2004)
https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

The difference in this implementation is that we are using cosine similarity
between tf-idf representations of sentences as a similarity measure between
sentences, rather than the word overlap measure they propose. This has been
found (https://arxiv.org/pdf/1602.03606.pdf) to offer better results.
"""

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from highlights.internals.graph import Graph
from highlights.internals.pagerank import pagerank_weighted
from highlights.internals.helpers import summary_length, NLP

def _textrank_scores(sentences):
    """ Given a list of sentences compute the importance score of each sentence
    using the weighted pagerank algorithm.

    Example:
        input: ['One sentence', 'Another sentence']
        output: {0: 0.120993, 1: 0.92830}

    Args:
        sentences: list of strings
    Returns:
        dictionary mapping list indices to importance score
    """
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
    """ Given a string, return a list holding the most important sentences in
    the given string text
    Args:
        text (string): text to be summarized
        len_func (func x: y): function taking the length in sentences of the
        given text and returning the number of sentences to be extracted
    Returns:
        a list holding the most important sentences sorted in decreasing order
        of importance
    """
    sentences = [sent.text for sent in NLP(text).sents]
    scores = _textrank_scores(sentences)

    sum_len = len_func(len(sentences))
    sent_scores = [(scores[i], s) for i, s in enumerate(sentences)]
    top_sentences = sorted(sent_scores, reverse=True)[:sum_len]

    return [s[1] for s in top_sentences]
