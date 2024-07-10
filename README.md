# Classifying News Articles with Different Models and Techniques


## Models From:

- **SciKit Learn**
  - Random Forest Classifier
  - MLP Classifier
- **HuggingFace**
  - DeBERTA
  - MiniLM

## Model Links:

- [DeBERTA](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
- [Mini LM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


## Model Accuracies:
 
These models were tested on 10,000 articles each. 

|          | SciKit Learn Models | DeBERTA-v3|Mini LM|
|----------|----------|----------|----------|
|**Top 3 Categories**|64.35%|50.70%|75.42% |
|**Top 4 Categories**|65.54%|48.50%|73.10% |
|**Top 5 Categories**|37.00%|36.60%|71.07% |


## Setup:

- Install [Python](https://www.python.org) version 3.8 or above
- Install Conda using `pip install conda`
- Install dependencies from the `environment.yml` file using `conda env create -f environment.yml`
- Activate environment using `conda activate environment.yml`
- Download the [dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) from Kaggle
- **For best results, choose** `MiniLM` **when running** `main.py`

## Example of Program Use:

![Alt Text](https://media.giphy.com/media/vKxz9P2LFuMjUuDnuP/giphy.gif)
