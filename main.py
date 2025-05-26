import time
from load_dataset import load_data
from preprocessing import apply_preprocessing
from vectorization import vectorize_text
from model import train_and_eval, save_model
from print_results import print_results

start = time.time()

# 1. Load dataset
train = load_data('aclImdb/train')
test = load_data('aclImdb/test')

# 2. Clear text
train = apply_preprocessing(train)
test = apply_preprocessing(test)

# 3. Vectorize text
X_train, X_test, y_train, y_test, vectorizer = vectorize_text(train, test)

# 4.Train and evaluate models
results = train_and_eval(X_train, X_test, y_train, y_test)

# Choose the best model based on accuracy
better_name = max(results, key=lambda k: results[k]["accuracy"])
better_model = results[better_name]["model"]

print(f"\n🧠 Better model: {better_name} ({results[better_name]['accuracy']:.4f})")
# Save the best model and vectorizer
save_model(better_model, vectorizer)

end = time.time()
print(f"Training and evaluate completed in {end - start:.2f} seconds")

print_results(results)



