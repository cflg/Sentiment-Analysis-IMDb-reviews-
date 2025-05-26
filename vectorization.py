from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(train, test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    X_test = vectorizer.transform(test['text'])
    y_train = train['label']
    y_test = test['label']
    return X_train, X_test, y_train, y_test, vectorizer
