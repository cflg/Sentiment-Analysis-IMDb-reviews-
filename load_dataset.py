import os
import pandas as pd

def load_data(path):
    data = []
    labels = []

    for etiqueta in ['pos', 'neg']:
        folder = os.path.join(path, etiqueta)
        print(f"Loading {etiqueta} from {folder}")
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                data.append(f.read())
                labels.append(1 if etiqueta == 'pos' else 0)
    print(f"Total loaded of {path}: {len(data)} reviews")
    return pd.DataFrame({'text': data, 'label': labels})

if __name__ == "__main__":
    train = load_data('aclImdb/train')
    test = load_data('aclImdb/test')
