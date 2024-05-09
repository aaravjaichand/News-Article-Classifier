import json

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

f = open("News_Category_Dataset_v3.json", "r")

categories = []
inputs = []
targets = []

i = 0

rAmount = input("Amount of lines to be read in: ")

for line in f:
    data = json.loads(line)
    input_ = [data["headline"], data["short_description"]]
    target = data["category"]
    inputs.append(input_)
    targets.append(target)

    i += 1
    if i == int(rAmount):
        break

for i in range(len(targets)):
    if targets[i] not in categories:
        categories.append(targets[i])

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
        return "Undefined"
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
        return "0"
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
        return "0"
    avgWL = totalNumLetters / totalWords
    return avgWL

def sentenceLen(string):
    return len(string.split())

def wordLookup(string, word):
    stringSplit = string.split()

    count = 0
    for i in range(len(stringSplit)):
        if stringSplit[i].lower() == word.lower():
            count += 1
    return count

for i in range(len(inputs)):
    print()
    # i+1, inputs[i][1], "AVERAGE WORD LENGTH:", avg(inputs[i][1]), "ARTICLES RATIO:" ,articles(inputs[i][1]),
    print(
        "Certain Word Occurance", wordLookup(inputs[i][1], "The"),
        "CATEGORY", targets[i]
    )
    if i == len(inputs) - 1:
        print()

def main():
    digits_set = datasets.load_digits()
    inputs = digits_set.data
    target = digits_set.target

    classifier = MLPClassifier(random_state=0)
    test_size = 10

    classifier.fit(inputs[test_size:], target[test_size:])

    results = classifier.predict(inputs[:test_size])

if __name__ == '__main__':
    main()
