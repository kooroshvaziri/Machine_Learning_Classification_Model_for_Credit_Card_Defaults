# Machine Learning Classification Model for Credit Card Defaults

The purpose of this project is to detect credit card defaults with machine learning classifiers and build an efficient classiier to identify bank customers who would default next month. It is a binary classifier problem, and the data has taken from ["Default of Credit Card Clients Data Set"](https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Default%20of%20Credit%20Card%20Clients) where itself comes from UC Irvine's [UCI Machine Learning Repo](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

# Jupyter Notebooks
The accompanied [Jupyter Notebook](capstone.ipynb) provides the exploratory data analysis, calculations, methodologies, and different classifier models performance comparisons.

# Exploratory Data Analysis and Data Cleaning
This dataset is relatively clean with no missing values and no duplicates. There were originally 30,000 rows and 24 features (excluding ID) in the dataset. Cleaning the data involved following steps:

1- ID column was dropped.
2- MARRIAGE column according to the legend should have had values between 1 and 3, but there were rows that has 0 in them. These rows were updated to 3 to represent OTHERS category.
3- EDUCATION column supposed to have values from 1-4, but we see values 0, 5, and 6. These values were all converted to OTHERS or 4.
4- BILL_AMT1 through BILL_AMT6 columns all had negative values. All rows were dropped who had negative values for any BILL_AMT columns.
5- LIMIT_BAL column is right-skewed with outliers as seen in below picture, so it was trimmed to balances under $600,000.

After clean up, the data has 22299 rows and 24 columns. The data is highly imbalanced with 77.3% non-default and only 22.7% default cases as seen in below image.

![Balance Distribution and Imbalance Data Labels](images/capstone_00.png)

Since we aim to classify and find default customers, this imbalance data poses some difficulties, and the data needs to be handled with case.






