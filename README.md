# Highlights

This is a Python 3 package for automatic summarization. You can use it to automatically generate article highlights/summaries. There are currently three extractive algorithms implemented:

- textrank [Mihalcea, 2004](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
- erank (an extended version of the textrank algorithm)
- tgraph (a variation of the algorithm presented in [Parveen, 2015](https://www.ijcai.org/Proceedings/15/Papers/187.pdf))

## The algorithms

**TextRank**

The algorithm implemented in this library differs from the one presented in the original paper in that it is using the cosine similarity measure between tf-idf representations of sentences rather than the word overlap measure they propose. This [paper](https://arxiv.org/pdf/1602.03606.pdf) has found the cosine measure to produce better results than the original.

**Extended TextRank**

The erank algorithm is much like textrank, the only difference being that the original graph is extended with nodes for article title, topics, and named entities. This adds extra information to the graph and produces better results on the dataset I have tested this on.

**TGRAPH**

This tgraph implementation follows much of the original paper, the only (major) difference being that a different function is used to compute sentence importance, namely, the erank function above. I have found this to produce better results than using the original graph and hits algorithm they propose.

Do note this performance comparison says little about their implemenation, only my own implementations and tests. What trully matters however is that this algorithm is the best among the ones currently implemented (also the slowest but probably usable in most cases).

## Installation

```Python
pip install highlights
python -m spacy download en
```

Note this package depends on and will install quite a few other packages:

- numpy, scipy, scikit-learn, for some numerical computations
- spacy, for sentence segmentation and named entity recognition
- gensim, for lda modeling
- pulp, for linear programming

Don't forget to download the "en" model for spacy after installing the package, as instructed above.

The package currently requires the most recent versions of these libraries. This is certanly way too restrictive and I will try and update the package as soon as possible to lower the version requirements.

## Usage

The textrank algorithm has the simplest API. Just pass it some text, and it will return the most important sentences in that text as a list of strings.

```Python
from highlights.extractive import textrank

most_important_sentences = textrank(text)
```

By default, the algorithm returns a number of sentences equal to 1/3 times the number of sentences in the given text, but no more than three. This is the function that computes that number:

```Python
def summary_length(n_of_sentences):
    return min(3, n_of_sentences // 3)
```

You can however pass your own function:

```Python
from highlights.extractive import textrank

most_important_sentences = textrank(text, len_func=lambda n_of_sentences: 3)
```

The erank and tgraph functions have slightly more complex APIs. They require a pretrained Gensim LDA model in order to compute the topics associated with each article.

```Python
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

from highlights.extractive import erank, tgraph
from highlights import make_topic_terms

# load lda stuff into memory
word_dict = Dictionary.load_from_text('/path/to/your_model_wordids.txt')
lda = LdaModel.load('/path/to/your_model.model')

# This is a dictionary mapping topic ids to top 10 words for that topic.
# Precomputing it here makes the algorithms run faster.
# This computation itself however can still take quite a while so
# it may be best to pickle this dictionary and load it from disk in future runs
topic_terms = make_topic_terms(lda)

# note these algorithms can also use the title of the article but this is optional
best_sentences_erank = erank(text, lda, word_dict, topic_terms, title)
best_sentences_tgraph = tgraph(text, lda, word_dict, topic_terms, title=None)

# just like with textrank, you can also pass a custom length function
erank(text, lda, word_dict, topic_terms, title, len_func: lambda x: 3)
tgraph(text, lda, word_dict, topic_terms, title=None, len_func: lambda x: 3)
```
