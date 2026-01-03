# Fake News Detection using Machine Learning

This project focuses on detecting fake news articles using machine learning and natural language processing (NLP) techniques. The model classifies news as Fake or Real based on textual content.

## Dataset Information

Dataset Source: Kaggle  
Dataset Name: Fake News Detection Dataset  
Dataset Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

The dataset files are larger than GitHub's 25 MB upload limit, so they are not included in this repository. Instead, the dataset must be downloaded directly from Kaggle using the link above.

## How to Download and Use the Dataset

1. Create or log in to your Kaggle account.
2. Open the dataset link provided above.
3. Download the dataset and extract the files.
4. Place the extracted CSV files in the same directory as the notebook, or update the file paths inside the notebook accordingly.

Example:
data = pd.read_csv("Fake.csv")

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Natural Language Processing (NLP)  
Jupyter Notebook  

## Machine Learning Approach

The project follows these steps:
- Data loading and exploration
- Text cleaning and preprocessing
- Feature extraction using TF-IDF Vectorization
- Training machine learning models such as Logistic Regression, Naive Bayes, and Passive Aggressive Classifier
- Evaluating models using accuracy and confusion matrix

## How to Run the Project

1. Clone the repository:
git clone https://github.com/Raghava001-web/Fake_news_detection_.git

2. Download the dataset from Kaggle using the provided link.

3. Open the notebook:
jupyter notebook Fake_News_Detection.ipynb

4. Run all cells sequentially.

## Results

The trained machine learning model is able to classify fake and real news articles with good accuracy after proper preprocessing and feature extraction.



This project is created for educational and academic purposes.
