import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from transformers import pipeline

from features import preposition, upperLower, articles, avg, sentenceLen

def display_accuracy(target, predictions, labels, plot_title):
    cm = confusion_matrix(target, predictions)
    unique_labels = np.unique(target)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()

def get_data():
    global acceptedCats
    global acceptedCats1
    acceptedCats = ["WORLD NEWS", "POLITICS", "ENTERTAINMENT"]
    acceptedCats1 = []
    saved_file = Path('saved_data.npz')
    list_of_strings = []

    i = 0
    for line in (list(open("News_Category_Dataset_v3.json", "r"))):
        data = json.loads(line)
        headline = data["headline"]
        short_desc = data["short_description"]
        category = data["category"]
        acceptedCats1.append(category)

        if category in acceptedCats:
            list_of_strings.append(headline + '. ' + short_desc)

        if i == 1:
            break
    acceptedCats1 = np.unique(acceptedCats1)
    if not saved_file.exists():
        inputs = []
        targets = []

        for line in tqdm(
                list(open("News_Category_Dataset_v3.json", "r")),
                desc='Loading json...'
        ):

            data = json.loads(line)
            category = data["category"]
            headline = data["headline"]
            short_desc = data["short_description"]
            if category in acceptedCats:
                inputs.append(headline + '. ' + short_desc)
                targets.append(category)

        feature_funcs = [preposition, upperLower, articles, avg, sentenceLen]
        inputs = np.array([
            [feature_func(inp) for feature_func in feature_funcs]
            for inp in tqdm(inputs, desc='Processing features...')
        ])
        targets = np.array(targets)
        np.savez(saved_file, inputs=inputs, targets=targets)
    else:
        arr = np.load(saved_file)
        inputs = arr['inputs']
        targets = arr['targets']

    return inputs, targets, list_of_strings

def sklearn_model():
    inputs, targets, list_of_strings = get_data()
    test_size = int(len(inputs) * 0.1)

    # Random Forest Classifier:
    m = RandomForestClassifier(
        random_state=12, n_estimators=70, max_depth=5, verbose=1)
    m.fit(inputs[test_size:], targets[test_size:])
    results = m.predict(inputs[:test_size])

    # MLP Classifier
    lrs = np.logspace(-4, -1, 4)
    accs = []
    i = 1
    for lr in lrs:
        classifier = MLPClassifier(
            random_state=1, hidden_layer_sizes=(10, 10, 50),
            learning_rate_init=lr, batch_size=test_size, max_iter=20, verbose=1
        )
        classifier.fit(inputs[test_size:], targets[test_size:])
        results = classifier.predict(inputs[:test_size])
        print(i, "of", len(lrs))
        i += 1
        acc = np.mean(results == targets[:test_size])
        accs.append(acc)
    plt.plot(lrs, accs)
    plt.show()

    optimalLR = lrs[accs.index(max(accs))]
    print("Best LR tested was", optimalLR)
    classifier = MLPClassifier(
        random_state=1, hidden_layer_sizes=(10, 10, 50),
        learning_rate_init=optimalLR, batch_size=test_size, max_iter=20,
        verbose=1
    )
    classifier.fit(inputs[test_size:], targets[test_size:])
    results = classifier.predict(inputs[:test_size])
    display_accuracy(
        targets[:test_size], results, np.unique(targets),
        "Confusion Matrix (Close to view accuracy)"
    )
    print("Min loss:", min(classifier.loss_curve_))

def deep_learning_model():

    model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )
    article = input("Enter article to be categorized: ")
    results = model(article, candidate_labels=acceptedCats1, verbose=0)

    scores = results["scores"]
    labs = results["labels"]

    print("Predicted Label:", labs[scores.index(max(scores))])

    # print(f'Accuracy: {np.mean(results == targets[:test_size])}')

def main():
    # sklearn_model()
    deep_learning_model()

if __name__ == '__main__':
    main()
