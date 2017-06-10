import spacy

NLP = spacy.load('en')


def summary_length(n_of_sentences):
    return min(3, n_of_sentences // 3)


def make_topic_terms(lda_model):
    topic_terms = {}
    for i in range(lda_model.num_topics):
        topic_terms[i] = lda_model.get_topic_terms(i, topn=10)

    return topic_terms
