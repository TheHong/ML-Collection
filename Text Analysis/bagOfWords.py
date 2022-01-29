from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def countDemo():
    print("\nUsing CountVectorizer")
    documents = [
        "The quick brown fox jumped over the lazy dog.",
        "你好，我是加拿大人", # Trying another language 
    ]

    vectorizer = CountVectorizer()
    vectorizer.fit(documents) # Tokenize and build vocab from the text
    print("Vocabulary", vectorizer.vocabulary_)  # Show what is tokenized
    encoding = vectorizer.transform(documents) # Encode some other text
    print("Shape of encoding", encoding.shape)
    print("Encoding type", type(encoding))
    print("Encoding as array", encoding.toarray())


def tfidfDemo():
    """ 
    TF-IDF = Term Frequency, Inverse Document frequency
    TF => Indicates how frequent in a certain doc
    IDF => Downscales words that are appear in other docs

    Takes into account words that are more "interesting", where interesting words 
    are those that are frequent in a certain doc, but not in other
    """

    print("\nUsing TfidfVectorizer")
    documents = [
        "The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"
    ]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents) # Tokenize and build vocab from the text
    print("Vocabulary", vectorizer.vocabulary_)
    print("Vocabulary", vectorizer.idf_)
    encoding = vectorizer.transform([documents[0]])
    print("Shape of encoding", encoding.shape)
    print("Encoding as array", encoding.toarray())


def hashDemo():
    print("\nUsing HashingVectorizer")
    documents = ["The quick brown fox jumped over the lazy dog."]

    vectorizer = HashingVectorizer(n_features=20)  # Smaller the number, the more likely collisions will happen
    encoding = vectorizer.transform(documents)
    print("Shape of encoding", encoding.shape)
    print("Encoding as array", encoding.toarray())  # Normalized word count between range of [-1, 1]


if __name__ == "__main__":
    countDemo()
    tfidfDemo()
    hashDemo()