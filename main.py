import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel

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

        file_lines = list(open("News_Category_Dataset_v3.json", "r"))
        for line in tqdm(file_lines, desc='Loading json...'):
            data = json.loads(line)
            headline = data["headline"]
            short_desc = data["short_description"]
            category = data["category"]
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
    test_size = 10000
    print(len(inputs))

    # Random Forest Classifier:
    # m = RandomForestClassifier(
    #     random_state=12, n_estimators=70, max_depth=5, verbose=1)
    # m.fit(inputs[test_size:], filteredTargets[test_size:])
    # results = m.predict(inputs[:test_size])

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
        print(acc)
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
    acc = np.mean(results == filteredTargets[:test_size])
    print("Accuracy of best LR:", acc)
    print("Min loss:", min(classifier.loss_curve_))
    print(results)

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

            start = i * batch_size
            end = start + batch_size

            if end > total:
                break

            results = model(
                filteredHeadlines[start:end], candidate_labels=acceptedCats, verbose=1)

            for j in range(batch_size):
                prediction = results[j]["labels"]

                if prediction[0] == filteredTargets[j]:
                    correct += 1
        return correct / total

    if modelToUse == 2:
        correct = 0
        total = num_batches * batch_size
        for i in range(num_batches):

            start = i * batch_size
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

    return correct // total

def directlyLoadedModel():
    global num_batches
    num_batches = int(input("Number of batches? "))
    def singleAcc(layer):
        inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets = get_data()

        batch_size = int(input("Batch size: "))

        correct = 0
            
        total = num_batches * batch_size

        
        model_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size

            if end > total:
                break

            with torch.inference_mode():
                headlines = filteredHeadlines[start:end]
                last_hidden_states = model(
                        **tokenizer(headlines, return_tensors='pt', padding=True, truncation=True),
                        output_hidden_states=True
                    ).hidden_states
                
                last_hidden_states_labels = model(**tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

            for j in range(batch_size):
                predictions = (last_hidden_states_labels[layer].mean(axis=1) @ last_hidden_states[layer].mean(axis=1).T).softmax(0).argmax(axis=0)
                prediction_strings = np.array(acceptedCats)[np.array(predictions)]
                targets = filteredTargets[start:end]
                acc = (prediction_strings == targets).mean() * 100

                if prediction_strings[j] == filteredTargets[j]:
                    correct += 1
        print(correct / total)
    
    def optLayer():
        inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets = get_data()


        bestAcc = -1
        bestLayer = -1
        listAccuracies = []


        batch_size = int(input("Batch Size: "))

        correct = 0

        listAccuracies = []
        bestLayer = -1
        bestAcc = -1
            
        total = num_batches * batch_size

        
        model_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        for l in tqdm(range(13)):
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                if end > total:
                    break

                with torch.inference_mode():
                    headlines = filteredHeadlines[start:end]
                    last_hidden_states = model(
                            **tokenizer(headlines, return_tensors='pt', padding=True, truncation=True),
                            output_hidden_states=True
                        ).hidden_states
                    
                    last_hidden_states_labels = model(**tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

                for j in range(batch_size):
                    predictions = (last_hidden_states_labels[l].mean(axis=1) @ last_hidden_states[l].mean(axis=1).T).softmax(0).argmax(axis=0)
                    prediction_strings = np.array(acceptedCats)[np.array(predictions)]
                    targets = filteredTargets[start:end]
                    

                    if prediction_strings[j] ==  targets[j]:
                        correct += 1
            acc = correct / total

            if acc > bestAcc:
                bestAcc = acc
                bestLayer = l

            listAccuracies.append(acc)
            correct = 0
            
        plt.xlabel("Layer used by model")
        plt.ylabel("Accuracy")
        plt.plot(listAccuracies)   
        plt.show()

        return bestLayer, bestAcc


    funcToUse = int(input("Would you like to find the optimal layer(1) to use or run on one layer(2)? My choice: "))
    
    if funcToUse == 2:
        singleAcc(int(input("What layer (0-12): ")))
    else:
        bestLayer, bestAcc = optLayer()

        print("Optimal layer:", bestLayer)
        print("Accuracy with optimal layer: ", bestAcc)

def sentence_transformers():
    inputs, allTargets, allHeadlines, allCategories, filteredHeadlines, filteredTargets = get_data()

    global samples
    samples = int(input("Number of samples? "))
    
    global headlines
    headlines = filteredHeadlines[: samples]

    def batches(layer):
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            last_hidden_states = model(**tokenizer(headlines, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states
            last_hidden_states_labels = model(**tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states


        predictions = (last_hidden_states_labels[layer].mean(axis=1) @ last_hidden_states[layer].mean(axis=1).T).softmax(0).argmax(axis=0)
        prediction_strings = np.array(acceptedCats)[np.array(predictions)]
        targets = filteredTargets[: samples]
        print(f'Accuracy: {(prediction_strings == targets).mean() * 100:.3f}%')


    def graph():
        with torch.inference_mode():
            
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            last_hidden_states = model(**tokenizer(headlines, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states
            last_hidden_states_labels = model(**tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

        bestLayer = -1
        bestAcc = -1

        listAccuracies = []

        for i in tqdm(range(len(last_hidden_states)), desc= "Finding optimal layer..."):
            predictions = (last_hidden_states_labels[i].mean(axis=1) @ last_hidden_states[i].mean(axis=1).T).softmax(0).argmax(axis=0)
            prediction_strings = np.array(acceptedCats)[np.array(predictions)]
            targets = filteredTargets[: samples]
            acc = (prediction_strings == targets).mean() * 100
            if acc > bestAcc:
                bestAcc = acc
                bestLayer = i
            listAccuracies.append(acc)
        plt.xlabel("Layer used by model")
        plt.ylabel("Accuracy")
        plt.plot(listAccuracies)   
        plt.show()

        return bestLayer, bestAcc
    
    funcToUse = int(input("Would you like to find the optimal layer(1) to use or run on one layer(2)? My choice: "))
    
    if funcToUse == 2:
        batches(int(input("What layer (0-7): ")))
    else:
        bestLayer, bestAcc = graph()
        print("Optimal layer:", bestLayer)
        print("Accuracy with optimal layer: ", bestAcc)

def plotAccs():
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

def main():
    global acceptedCats
    acceptedCats = []

    fpath = Path("saved_data.npz")

    if fpath.exists():
        fpath.unlink()
    print("\n")
    userChoice = int(input(
        "Would you like to classify one singular news article (1), or run the program on a group of articles(2)?: "))

    if userChoice == 1:

        articleHeadline = input(
            "Copy and paste article headline and short description here: ")

        with torch.inference_mode():
            acceptedCats = ["POLITICS", "WELLNESS",
                            "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY"]

            model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2")
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2")
            last_hidden_states = model(**tokenizer(articleHeadline, return_tensors='pt',
                                       padding=True, truncation=True), output_hidden_states=True).hidden_states
        
            last_hidden_states_labels = model(
                **tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

            predictions = (last_hidden_states_labels[-1].mean(
                axis=1) @ last_hidden_states[-1].mean(axis=1).T).softmax(0).argmax(axis=0)
            prediction1 = np.array(acceptedCats)[np.array(predictions)]


            acceptedCats = ['ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE', 'COMEDY',
                'CRIME', 'CULTURE & ARTS', 'DIVORCE', 'EDUCATION', 'ENTERTAINMENT',
                'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK', 'GOOD NEWS', 'GREEN', 'HEALTHY LIVING',
                'HOME & LIVING', 'IMPACT', 'LATINO VOICES', 'MEDIA', 'MONEY', 'PARENTING',
                'PARENTS', 'POLITICS', 'QUEER VOICES', 'RELIGION', 'SCIENCE', 'SPORTS', 'STYLE',
                'STYLE & BEAUTY', 'TASTE', 'TECH', 'THE WORLDPOST', 'TRAVEL', 'U.S. NEWS',
                'WEDDINGS', 'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS', 'WORLDPOST']

            last_hidden_states_labels = model(
                **tokenizer(acceptedCats, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

            predictions = (last_hidden_states_labels[-1].mean(
                axis=1) @ last_hidden_states[-1].mean(axis=1).T).softmax(0).argmax(axis=0)
            prediction2 = np.array(acceptedCats)[np.array(predictions)]

            print("\n")
            
            if prediction1 == prediction2:
                print("Predicted: ", prediction1[0])
            else:
                print("Predicted: ", prediction1[0] , "and", prediction2[0])
            print("\n")

    else:
        numCategories = int(
            input("Number of Categories: (1) Top 3; (2) Top 4; (3) Top 5: "))
        if numCategories == 1:
            acceptedCats = ["POLITICS", "WELLNESS", "ENTERTAINMENT"]
        elif numCategories == 2:
            acceptedCats = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL"]
        else:
            acceptedCats = ["POLITICS", "WELLNESS",
                            "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY"]

        modelToUse = int(input(
            "Which model would you like to use? (1) SciKit Learn; (2) DeBERTA-v3; (3) Sentence Transfomers: "))

        if modelToUse == 1:
            sklearn_model()
        elif modelToUse == 2:
            directlyLoadedModel()
        else:
            sentence_transformers()

if __name__ == '__main__':
    main()
