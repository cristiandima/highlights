import spacy

from collections import defaultdict

from pulp import *
from sklearn.feature_extraction.text import TfidfVectorizer

from highlights.extractive.erank import _erank_scores, _get_topics, _get_named_entities
from highlights.internals.helpers import summary_length, NLP

_word_tokenize = TfidfVectorizer(stop_words='english').build_analyzer()


def _sent_topics(sentences, topic_words):
    sent_topics = {}
    topic_words = [set(tw) for tw in topic_words]

    for s_i, sentence in enumerate(sentences):
        sent_topics[s_i] = set()
        words = set(_word_tokenize(sentence))
        for t_i, topic in enumerate(topic_words):
            if len(topic.intersection(words)) > 0:
                sent_topics[s_i].add(t_i)

    return sent_topics


def _topic_sents(sent_topics):
    topic_sents = defaultdict(set)
    for sent_key in sent_topics:
        for topic_key in sent_topics[sent_key]:
            topic_sents[topic_key].add(sent_key)

    return topic_sents

def _coh_scores(sent_topics):
    coh_score = {}
    sent_keys = sent_topics.keys()
    for this_sent in range(len(sent_keys)):
        score = 0
        this_topics = sent_topics[this_sent]
        for that_sent in range(this_sent + 1, len(sent_keys)):
            that_topics = sent_topics[that_sent]
            score += len(this_topics.intersection(that_topics))

        coh_score[this_sent] = score

    total = sum(coh_score.values())
    if total != 0:
        for k in coh_score:
            coh_score[k] = coh_score[k] / total

    return coh_score


def tgraph(text, lda, word_dict, topic_terms, title=None, len_func=summary_length):
    nlp_doc = NLP(text)
    sentences = [sent.text for sent in nlp_doc.sents]

    topics = _get_topics(nlp_doc, lda, word_dict, topic_terms)
    named_entities = _get_named_entities(nlp_doc)
    scores = _erank_scores(nlp_doc, topics, named_entities, title)

    sent_topics = _sent_topics(sentences, topics)
    topic_sents = _topic_sents(sent_topics)
    coh_scores = _coh_scores(sent_topics)

    # linear program
    prob = LpProblem('TGRAPH', LpMaximize)

    # binary variables
    sent_vars = {}
    for i in sent_topics:
        sent_vars[i] = LpVariable('s'+str(i), 0, 1, LpInteger)

    topic_vars = {}
    for i in topic_sents:
        topic_vars[i] = LpVariable('t'+str(i), 0, 1, LpInteger)

    # ojective function
    obj = LpAffineExpression()
    for s_key, s_var in sorted(sent_vars.items()):
        obj = obj + (coh_scores[s_key] + scores[s_key]) * s_var

    for t_key, t_var in sorted(topic_vars.items()):
        obj = obj + t_var

    prob += obj, 'tgraph summarization problem'

    # constraints
    # summary length
    max_len = LpAffineExpression()
    for s_key, s_var in sorted(sent_vars.items()):
        max_len += s_var
    max_len = max_len <= len_func(len(sentences))

    prob += max_len, 'summary length'
    # eq 6 (topics in a sentence are selected when sentence is selected)
    for s_key in sent_vars:
        exp = LpAffineExpression()
        topic_num = len(sent_topics[s_key])
        if topic_num == 0:
            continue
        for topic_key in sent_topics[s_key]:
            exp = exp + topic_vars[topic_key]

        exp = exp >= topic_num * sent_vars[s_key]
        prob += exp, 'topics in ' + str(s_key) + ' must be selected'

    #eq 7 (sentences containing selected topics are selected as well)
    for t_key in topic_vars:
        exp = LpAffineExpression()
        sent_num = len(topic_sents[t_key])
        if sent_num == 0:
            continue
        for sent in topic_sents[t_key]:
            exp = exp + sent_vars[sent]

        exp = exp >= topic_vars[t_key]
        prob += exp, 'sentences containing topic ' + str(t_key) + ' must be selected'

    prob.solve()

    summary = []
    s_vars = [v for v in prob.variables() if v.varValue != 0 and 't' not in v.name]
    s_vars = [int(v.name[1:]) for v in s_vars]
    for v in sorted(s_vars):
        summary.append(sentences[v])

    return summary
