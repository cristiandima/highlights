import spacy

NLP = spacy.load('en')


def summary_length(n_of_sentences):
    """ Given the length of a document in sentences this function will return
    an acceptable summary length
    """
    return min(3, n_of_sentences // 3)


def make_topic_terms(lda_model):
    """ Produces a {topic_id -> [ten most important words]} mapping for a given
    gensim lda model.

    These topic terms are used a lot inside some of the algorithms and
    calling get_topic_terms on the lda model everytime is quite slow. Computing
    this mapping once and reusing it across calls greatly improves performance.

    Args:
        lda_model: Gensim lda model

    Returns:
        a dictionary mapping topic ids to lists of the 10 most important words
        for each topic
    """
    topic_terms = {}
    for i in range(lda_model.num_topics):
        topic_terms[i] = lda_model.get_topic_terms(i, topn=10)

    return topic_terms
