# Fake News Detection Using Machine Learning

A student-level machine learning project that classifies news text as **Fake** or **Real** using TF-IDF vectorization and Logistic Regression. A simple Streamlit interface lets users enter a headline or article and view the model's prediction scores.

## Technologies

- Python
- pandas
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Project structure

```text
fake-news-detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.txt
└── model/
    └── README.txt
```

## Dataset setup

This project expects two CSV files:

- `data/Fake.csv`
- `data/True.csv`

The commonly used Kaggle Fake and Real News dataset has these filenames and a `text` column. The training script can also combine `title` and `text` if both columns are present.

**Dataset is not included in this repository.** Add the dataset locally after checking its license and terms. Large CSV files and trained model files are excluded from GitHub.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install packages:
```bash
python -m pip install -r requirements.txt
```

## Train the model

Make sure both CSV files are inside `data/`, then run:

```bash
python train_model.py
```

The script:
1. Loads and combines the fake and real news datasets.
2. Cleans the text.
3. Splits the data into training and testing sets.
4. Converts text into TF-IDF features.
5. Trains Logistic Regression.
6. Prints accuracy, classification report, and confusion matrix.
7. Saves the trained model and vectorizer in `model/`.

## Run the Streamlit app

```bash
python -m streamlit run app.py
```

Paste a news article or headline into the text box and select **Analyze News**.

## Important limitations

This is an educational text-classification project, not a fact-checking service. It learns patterns from its training dataset and does not independently verify claims against trusted sources. Dataset bias, outdated examples, misleading writing, satire, and new topics can lead to incorrect predictions. The displayed percentages are model scores and should not be interpreted as certainty.

## Author

Muhammad Saad Javed  
BS Artificial Intelligence student

Update this README with your own screenshots, results, and improvements after you run and test the project.
