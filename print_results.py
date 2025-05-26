import matplotlib.pyplot as plt

def print_results(results):
    names = list(results.keys())
    accuracies = [r["accuracy"] for r in results.values()]
    times = [r["time"] for r in results.values()]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.barh(names, accuracies)
    plt.title("Accuracy by model")
    plt.xlabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.barh(names, times)
    plt.title("Training time by model")
    plt.xlabel("Seconds")

    plt.tight_layout()
    plt.show()
