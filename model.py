from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def train_and_eval(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Multinomial NB": MultinomialNB(),
        "Linear SVC": LinearSVC(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=10)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end = time.time()

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "time": end - start
        }

    return results

def search_better_parameters(X_train, y_train):
    model = LogisticRegression(solver='saga', max_iter=1000)
    parameters = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2', 'l1']
    }
    grid = GridSearchCV(model, parameters, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Better combination:", grid.best_params_)
    print("Better accuracy:", grid.best_score_)

    return grid.best_estimator_

def save_model(model, vectorizer, path_model='model.joblib', path_vectorizer='vectorizer.joblib'):
    joblib.dump(model, path_model)
    joblib.dump(vectorizer, path_vectorizer)