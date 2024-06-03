import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from transformers import pipeline


def preposition(string):
    prepositions = [
        "in", "on", "at", "of", "to", "with", "by", "for", "about", "from",
        "into", "onto", "upon", "within", "without", "through", "during",
        "among", "between", "above", "below", "under", "over", "underneath",
        "around", "near", "after", "before", "behind", "beside", "inside",
        "outside", "since", "until", "upon"
    ]

    prepositionCount = 0
    sentenceLength = len(string.split())
    sentence = string.split()
    for word in sentence:
        if word.lower() in prepositions:
            prepositionCount += 1

    if prepositionCount == 0 or sentenceLength == 0:
        return 0
    else:
        return prepositionCount / sentenceLength


def upperLower(string):
    upper = 0
    lower = 0
    for word in string:
        letters = list(word)
        for letter in letters:
            if letter.isupper():
                upper += 1
            else:
                lower += 1
    if lower == 0 or upper == 0:
        return 0
    
    return upper / lower


def articles(string):
    wordList = string.split()
    the_a_count = 0
    sentenceWC = len(wordList)
    the_a_count += wordList.count("The") + wordList.count("the") + \
        wordList.count("A") + wordList.count("a")
    if the_a_count == 0 or sentenceWC == 0:
        return 0
    
    articleRatio = the_a_count / sentenceWC

    return articleRatio


def avg(string):
    wordList = string.split()
    totalNumLetters = 0
    totalWords = len(wordList)
    for i in range(len(wordList)):
        word = wordList[i]
        totalNumLetters += len(word)
    if totalWords == 0 or totalNumLetters == 0:
        return 0
    avgWL = totalNumLetters / totalWords
    return avgWL


def sentenceLen(string):
    return len(string.split())
acceptedCats = ["WORLD NEWS", "POLITICS", "ENTERTAINMENT"]
def get_data():
    saved_file = Path('saved_data.npz')


    if not saved_file.exists():
        inputs = []
        targets = []
        list_of_strings = []
        for line in tqdm(
                list(open("News_Category_Dataset_v3.json", "r")),
                desc='Loading json...'
        ):
            
            data = json.loads(line)
            category = data["category"]
            headline = data["headline"]
            short_desc = data["short_description"]
            if category in acceptedCats:
                list_of_strings.append(headline + '. ' +short_desc)
                inputs.append(headline + '. ' +short_desc)
                targets.append(category)


        feature_funcs = [preposition, upperLower, articles, avg, sentenceLen]
        inputs = np.array([
            [feature_func(inp) for feature_func in feature_funcs]
            for inp in tqdm(inputs, desc='Processing features...')
        ])
        targets = np.array(targets)
        np.savez(saved_file, inputs=inputs, targets=targets, list_of_strings=list_of_strings)
    else:
        arr = np.load(saved_file)
        inputs = arr['inputs']
        targets = arr['targets']
        list_of_strings = arr["list_of_strings"]
    return inputs, targets, list_of_strings


def display_accuracy(target, predictions, labels, plot_title):
    cm = confusion_matrix(target, predictions)
    unique_labels = np.unique(target)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()
def main():
    inputs, targets, list_of_strings = get_data()
    test_size = int(len(inputs) * 0.1)

    # Random Forest Classifier:
    # m = RandomForestClassifier(
    #     random_state=12, n_estimators=70, max_depth=5, verbose=1)
    # m.fit(inputs[test_size:], targets[test_size:])
    # results = m.predict(inputs[:test_size])

    # MLP Classifier
    # lrs = np.logspace(-4, -1, 4)
    # accs = []
    # i = 1
    # for lr in lrs:
    #     classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(
    #     10, 10, 50), learning_rate_init=lr, batch_size=test_size, max_iter=20, verbose=1)
    #     classifier.fit(inputs[test_size:], targets[test_size:])


    #     results = classifier.predict(inputs[:test_size])
    #     print(i, "of", len(lrs))
    #     i+=1
    #     acc = np.mean(results == targets[:test_size])
    #     accs.append(acc)
    # plt.plot(lrs, accs)
    # plt.show()

    # optimalLR = lrs[accs.index(max(accs))]
    # print("Best LR tested was", optimalLR)
    # classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(
    #     10, 10, 50), learning_rate_init=optimalLR, batch_size=test_size, max_iter=20, verbose=1)
    # classifier.fit(inputs[test_size:], targets[test_size:])
    # results = classifier.predict(inputs[:test_size])
    # display_accuracy(targets[:test_size], results, np.unique(targets), "Confusion Matrix (Close to view accuracy)")
    # print("Min loss:", min(classifier.loss_curve_))
    

    model = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    results = model.predict(list_of_strings, labels=np.unique(targets))

    
    
    print(f'Accuracy: {np.mean(results == targets[:test_size])}')


if __name__ == '__main__':
    main()
