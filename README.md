# Machine Learning to train on news articles and descriptions

- Install [Python](https://www.python.org) version 3.8 or above
- Install Conda using `pip install conda`
- Install dependencies from the environment.yml file using `conda env create -f environment.yaml`
- Activate environment using `conda activate environment.yml`
- Download the [dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data)
- **For best results, choose** `MiniLM` **when running** `main.py`





**Model Links**:

- [DeBERTA](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
- [Mini LM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


**Accuracies**:

These models were tested on 10,000 articles each. 

|          | SciKit Learn Models | DeBERTA-v3|Mini LM|
|----------|----------|----------|----------|
|**Top 3 Categories**|64.35%|44.00%|75.42% |
|**Top 4 Categories**|65.54%|40.40%|73.10% |
|**Top 5 Categories**|50.98%|36.60%|71.07% |

