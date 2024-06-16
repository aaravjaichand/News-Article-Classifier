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

acceptedCats = ["POLITICS", "WELLNESS", "ENTERTAINMENT"]


def display_accuracy(target, predictions, labels, plot_title):
    cm = confusion_matrix(target, predictions)
    unique_labels = np.unique(target)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()


def get_data():
    allCategories = []
    allHeadlines = []
    allTargets = []
    filteredHeadlines = []
    filteredTargets = []



    for line in (list(open("News_Category_Dataset_v3.json", "r"))):
        data = json.loads(line)
        headline = data["headline"]
        short_desc = data["short_description"]
        category = data["category"]
        allCategories.append(category)
        allHeadlines.append([headline + " " + short_desc])


        if category in acceptedCats:
            filteredHeadlines.append(headline + short_desc)
            filteredTargets.append(category)

    allTargets = allCategories
    allCategories = np.unique(allCategories)

    saved_file = Path('saved_data.npz')
    if not saved_file.exists():
        inputs = []


        for line in tqdm(
                list(open("News_Category_Dataset_v3.json", "r")),
                desc='Loading json...'
        ):

            data = json.loads(line)
            headline = data["headline"]
            short_desc = data["short_description"]
            if category in acceptedCats:
                inputs.append(headline + '. ' + short_desc)


        feature_funcs = [preposition, upperLower, articles, avg, sentenceLen]
        inputs = np.array([
            [feature_func(inp) for feature_func in feature_funcs]
            for inp in tqdm(inputs, desc='Processing features...')
        ])
        filteredTargets = np.array(filteredTargets)
        np.savez(saved_file, inputs=inputs, filteredTargets=filteredTargets)
    else:
        arr = np.load(saved_file)
        inputs = arr['inputs']

    return inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets

def sklearn_model():
    inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets = get_data()
    test_size = int(len(inputs) * 0.1)

    # Random Forest Classifier:
    m = RandomForestClassifier(
        random_state=12, n_estimators=70, max_depth=5, verbose=1)
    m.fit(inputs[test_size:], filteredTargets[test_size:])
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
        classifier.fit(inputs[test_size:], filteredTargets[test_size:])
        results = classifier.predict(inputs[:test_size])
        print(i, "of", len(lrs))
        i += 1
        acc = np.mean(results == filteredTargets[:test_size])
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
    classifier.fit(inputs[test_size:], filteredTargets[test_size:])
    results = classifier.predict(inputs[:test_size])
    display_accuracy(
        filteredTargets[:test_size], results, np.unique(filteredTargets),
        "Confusion Matrix (Close to view accuracy)"
    )
    print("Min loss:", min(classifier.loss_curve_))

def deep_learning_model():
    inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets = get_data()

    model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

    modelToUse = 1

    num_batches = 17
    batch_size = 30

    if modelToUse == 1:
        correct = 0
        total = num_batches * batch_size
        for i in tqdm(range(num_batches)):

            start = i*batch_size
            end = start + batch_size

            if end > total:
                break

            results = model(
                filteredHeadlines[start:end], candidate_labels=acceptedCats, verbose=1)

            for j in range(batch_size):
                prediction = results[j]["labels"]

                if prediction[0] == filteredTargets[j]:
                    correct += 1
        return correct/total

    if modelToUse == 2:
        correct = 0
        total = num_batches * batch_size
        for i in range(num_batches):

            start = i*batch_size
            end = start + batch_size

            if end > total:
                break

            results = model(allHeadlines[start:end],
                            candidate_labels=allCategories, verbose=1)

            for j in range(batch_size):
                prediction = results[j]["labels"]

                if prediction[0] == allTargets[j]:
                    correct += 1
    if "TRAVEL" in filteredTargets:
        print("CHECK 1")
    if "STYLE & BEAUTY" in filteredTargets:
        print("CHECK 2")

    return correct//total

def main():
    accuracies = []

    accuracy = deep_learning_model()
    accuracies.append(accuracy)
    print(accuracy)

    acceptedCats.append("TRAVEL")
    accuracy = deep_learning_model()
    accuracies.append(accuracy)
    print(accuracy)

    

    acceptedCats.append("STYLE & BEAUTY")
    accuracy = deep_learning_model()
    accuracies.append(accuracy)
    print(accuracy)

    
    plt.plot([3, 4, 5], accuracies)
    plt.show()
    # listAccuracy = []
    # listTopCategories = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY"]
    
    # listAccuracy.append(deep_learning_model())

    # acceptedCats.append("TRAVEL")
    # listAccuracy.append(deep_learning_model())

    # acceptedCats.append("STYLE & BEAUTY")
    # listAccuracy.append(deep_learning_model())

    # plt.plot([3, 4, 5], listAccuracy)
    # plt.show()

if __name__ == '__main__':
    main()