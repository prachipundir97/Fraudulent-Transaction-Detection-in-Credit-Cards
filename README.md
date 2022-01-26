# Fraudulent-Transaction-Detection-in-Credit-Cards

This project contains the steps to arriving at a model for successfully detecting fraudulent transactions in credit card data.
In this project, the data massaging and pre-processing were completed. It was found that the features for analysis could be increased by One-Hot Encoding of dataset. Further, the dataset was split into five chunks of data, and a comparative analysis was performed for different Machine Learning classification models. Finally applying best ensemble technique, which is XGBoost with n_features=200 and ADASYN with decision tree, we were able to achieve high precision score and f1 score over all other classifiers.
In this project, we created a high-precision machine learning model to detect credit card frauds. Whenever a fraud is detected, human intervention is needed to verify whether the transaction is legitimate by calling the customer. This intervention can be reduced by a high precision model, as the number of false positives will be reduced.


## Installation

To run the project, the following libraries are needed:
xgboost library
imblearn library

Installation of xgboost can be done as follows:

```bash
pip install xgboost
```

Additionally, it can be easily installed in an anaconda environment with the below command:

```bash
conda install -c anaconda py-xgboost
```


Installation of imblearn can be done as follows:

```bash
pip install -U imbalanced-learn
```

Additionally, it can be easily installed in an anaconda environment with the below command:

```bash
conda install -c conda-forge imbalanced-learn
```


## Dataset

The project references the Tabformer Credit Card dataset

The below link is for downloading the transactions.tgz file for the dataset:
https://github.com/IBM/TabFormer/tree/main/data/credit_card

It can optionally be downloaded from:
https://ibm.box.com/v/tabformer-data

To unpack the TAR file,

Mac/Unix:

```bash
tar -xvf transactions.tgz
```

Windows:

Any popular archive tool such as WinZip/WinRAR can be used to extract the contents.

After unpacking, the following file is obtained:
card_transaction.v1.csv

Rename the file to 'card_fraud.csv' and add it to the Datasets directory of the project.


## Usage

1. Load all the notebooks into Jupyter and load the dataset into a separate folder called 'Datasets' which should be in the same directory as the Notebooks. All the other data files needed for the application will be populated by the notebooks themselves.

2. To execute the project, The notebooks must be run in the following order:
Exploratory_Analysis (Runtime: approx 20mins) -> Model_Analysis (Runtime: approx 15mins) -> Model_Enhancement (Runtime: approx 50mins)  -> Model_Final_Metrics (Runtime: approx 45mins)

3. There should ideally be no memory issues in these notebooks, just that they might take more time than the approx time specified above. In case memory issues are seen, after running each notebook, make sure to shutdown the notebook and close it. If needed, restart Jupyter/Anaconda.

4. If run correctly, the following data files will be populated in the Datasets folder after executing the first notebook:
	5 sampled_cc_n.csv files where n is in range [1,5]
	5 ohe_sampled_cc_n.csv files where n is in range [1,5] 
	
5. In case project encounters a memory issue before the One-Hot-Encoding step (last cell of the Exploratory_Analysis notebook), run the first and the second last cell of the notebook. Then run the last cell by uncommenting the following lines:
```bash
#df1 = pd.read_csv('Datasets/sampled_cc_1.csv')
#df2 = pd.read_csv('Datasets/sampled_cc_2.csv')
#df3 = pd.read_csv('Datasets/sampled_cc_3.csv')
#df4 = pd.read_csv('Datasets/sampled_cc_4.csv')
#df5 = pd.read_csv('Datasets/sampled_cc_5.csv')
```

and commenting the following lines:
```bash
del df
del backup
```

optionally, you can replace the last cell with the contents below and execute just that cell:

```bash
import numpy as np
import pandas as pd

def OHEForDF(input_df):
    ohe = preprocessing.OneHotEncoder(dtype=int, sparse=False, handle_unknown="ignore")


    data = ohe.fit_transform(input_df[["Merchant_State", "Use_Chip", "Card"]])

    input_df.drop(["Merchant_State", "Use_Chip", "Card"], axis=1, inplace=True)
    cats = pd.DataFrame(data, columns=ohe.get_feature_names())
    input_df = pd.concat([cats, input_df], axis=1)
    redundant_rows = len(input_df) - len(input_df)
    input_df.drop(input_df.tail(redundant_rows).index,inplace=True) # drop last n rows
    input_df.columns = [c.replace("x0_", "Merchant_State=").replace("x1_","Use_Chip=").replace("x2_","Card=") for c in input_df.columns]
    #input_df = addAllCols(input_df)
    return input_df

df1 = pd.read_csv('Datasets/sampled_cc_1.csv')
df2 = pd.read_csv('Datasets/sampled_cc_2.csv')
df3 = pd.read_csv('Datasets/sampled_cc_3.csv')
df4 = pd.read_csv('Datasets/sampled_cc_4.csv')
df5 = pd.read_csv('Datasets/sampled_cc_5.csv')
dfs = [df1, df2, df3, df4, df5]
i=1
for currDF in dfs:
    OHEForDF(currDF).to_csv('Datasets/ohe_sampled_cc_' + str(i) + '.csv', sep=',', encoding='utf-8')
    i += 1
    print("done")
```

## Author Attribution
This readme and all the project files have been created by:
Prachi Pundir, Manas Shukla & Anupa Shah of Virginia Polytechnic Institute and State University as a part of course CS_5644: Machine Learning With Big Data.
