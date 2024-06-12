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
    the_a_count += (
        wordList.count("The") + wordList.count("the")
        + wordList.count("A") + wordList.count("a")
    )
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
