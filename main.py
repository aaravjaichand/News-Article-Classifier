import json

import numpy as np
from sklearn.neural_network import MLPClassifier

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
    return upper / lower

def articles(string):
    wordList = string.split()
    the_a_count = 0
    sentenceWC = len(wordList)
    the_a_count += wordList.count("The") + wordList.count("the") + wordList.count("A") + wordList.count("a")
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

def main():
    str_inputs = []
    targets = []

    rAmount = int(input("Amount of lines to be read in: "))
    test_size = int(rAmount * .1)

    for i, line in enumerate(open("News_Category_Dataset_v3.json", "r")):
        data = json.loads(line)
        str_inputs.append(data["headline"] + data["short_description"])
        targets.append(data["category"])
        if i == rAmount:
            break

    feature_funcs = [preposition, upperLower, articles, avg, sentenceLen]
    inputs = np.array([
        [feature_func(inp) for feature_func in feature_funcs]
        for inp in str_inputs
    ])
    targets = np.array(targets)
    classifier = MLPClassifier(random_state=0)
    classifier.fit(inputs[test_size:], targets[test_size:])
    results = classifier.predict(inputs[:test_size])

    print(f'Accuracy: {np.mean(results == targets[:test_size])}')

if __name__ == '__main__':
    main()
