"""
This is in many ways identical to the textrank algorithms. The only difference
is that we expand the sentence graph to also include the title of the text,
the topics associated with the text, and the named entitites present

The output is still an importance score for each sentence in the original text
but these new nodes offer extra information and increase the weights of those
sentences which are more closely related to the topics/title/named entities
associated with the text
"""

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from highlights.extractive.textrank import _textrank_scores
from highlights.internals.helpers import summary_length, NLP

_word_tokenize = TfidfVectorizer(stop_words='english').build_analyzer()


def _get_named_entities(nlp_doc):
    """ Given a spacy document return the top ten most frequent name entities
    present in the text. Name entities appearing only once are skipped.

    Args:
        nlp_doc (spacy document): document to extract named entities from
    Returns:
        a list of words, the most frequent named entities present in the document
    """
    ignored_ents = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}
    ne = [n.text for n in nlp_doc.ents if n.label_ not in ignored_ents]
    ne = [n.replace('the', '').strip() for n in ne]
    ne = set(ne)

    counter = CountVectorizer(ngram_range=(1,2))
    counts = counter.fit_transform([nlp_doc.text])
    ne_scores = []
    for entity in ne:
        entity = entity.lower()
        if entity in counter.vocabulary_:
            ne_scores.append((counts[0, counter.vocabulary_.get(entity)], entity))

    ne_scores = sorted([n for n in ne_scores if n[0] != 1], reverse=True)[:10]
    return [n[1] for n in ne_scores]


def _get_topics(nlp_doc, lda, word_dict, topic_terms):
    """ Given a spacy document, as well as an lda model, this function returns
    a list of lists where each list holds the string words associated with each
    topic associated with the document
    """
    doc_bow = word_dict.doc2bow(_word_tokenize(nlp_doc.text))
    topics = lda.get_document_topics(doc_bow)

    topics_as_words = []
    for topic_tuple in topics:
        topic_words = []
        for word_tuple in topic_terms[topic_tuple[0]]:
            topic_words.append(word_dict[word_tuple[0]])

        topics_as_words.append(topic_words)

    return topics_as_words


def _erank_scores(nlp_doc, topics, named_entities, title=None):
    sentences = [sent.text for sent in nlp_doc.sents]
    original_len = len(sentences)

    for topic_words in topics:
        sentences.append(' '.join(topic_words))

    if len(named_entities) >= 1:
        sentences.append(' '.join(named_entities))

    if title is not None:
        sentences.append(' '.join(_word_tokenize(title)))

    scores = _textrank_scores(sentences)
    scores = {i: scores.get(i, 0) for i in range(original_len)}
    return scores


def erank(text, lda, word_dict, topic_terms, title=None, len_func=summary_length):
    nlp_doc = NLP(text)
    sentences = [sent.text for sent in nlp_doc.sents]

    topics = _get_topics(nlp_doc, lda, word_dict, topic_terms)
    named_entities = _get_named_entities(nlp_doc)
    scores = _erank_scores(nlp_doc, topics, named_entities, title)

    sum_len = len_func(len(scores))
    sent_scores = [(scores[i], s) for i, s in enumerate(sentences)]
    top_sentences = sorted(sent_scores, reverse=True)[:sum_len]

    return [s[1] for s in top_sentences]
