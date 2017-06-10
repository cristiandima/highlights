import collections

RougeScore = collections.namedtuple('RougeScore', ['recall', 'precision', 'f1'])

def generate_ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)


def counter_overlap(model_counter, reference_counter):
    overlap_sum = 0
    for ngram, count in model_counter.items():
        overlap_sum += min(count, reference_counter[ngram])

    return overlap_sum


def rouge_n(model, reference, n=2):
    if len(model) == 0:
        return RougeScore(recall=0, precision=0, f1=0)

    model_ngrams = generate_ngrams(model, n)
    ref_ngrams = generate_ngrams(reference, n)

    model_counter = collections.Counter(model_ngrams)
    ref_counter = collections.Counter(ref_ngrams)

    ref_total_count = sum(ref_counter.values())
    overlap_count = counter_overlap(model_counter, ref_counter)
    model_total_count = sum(model_counter.values())

    recall = overlap_count / ref_total_count
    precision = overlap_count / model_total_count
    f1 = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return RougeScore(recall=recall, precision=precision, f1=f1)
