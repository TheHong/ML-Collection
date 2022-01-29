# Text Analysis

## Bag-of-Words
- We need a way to convert text into something that ML algorithms could use as input.
- Bag-of-Words is one simple way to model text documents.

Bag-of-Words focuses on occurance of words and ignores order. The idea is to look all the words (i.e. tokens) that exist in known documents, then to encode a specific document, for each known word we look at how often each word appears in the document. 

We could also use groups of words instead of single words when we encode text. With this in mind, we are using "grams" when we encode text, where grams can be one token or multiple. So, we can have different combos, such as bigrams (2-gram) (e.g. "I am", "am Canadian", etc.) and trigrams (3-gram) (e.g. "I am Canadian", etc.). In conclusion, we can specify a certain Bag-of-Words model to be a certain **N-gram model**.