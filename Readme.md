## Introduction
Welcome to the Sreality anomaly detector - ML based tool that is supposed to search for the flats
that are underpriced with comparison to the similar ones.

## How does it work
1. With usage of the Sreality's API, information of all currently available flats are requested, processed to table format and saved. In the current version, only '2+1' and '2+kk' flats are retrieved.
2. Extracted data are afterwards used for training of ML model, where price of the flats is used as the target variable. Light GBM is selected as the used ML algorithms for various reasons: 
    * Still provides state of the art performance for tabular data for both tasks - Classification and Regression
    * Is tree based method - can easily discover interaction between independent features.
    * Is nonlinear method - does not require linear relationship between target and feature, therefore does not require much of data preprocessing
    * Can handle categorical variables implicitly (no one hot encoding or other method is necessary)
    * Still relatively easily explainable with for Example Shap method (so far not implemented).

   Cross validation is used for obtaining of optimal number of trees and therefore model would not be nor underfitting or overfitting. With this optimal number of trees, final model with all data is fit. Note: Script, that was used for feature selection is not enclosed. Feature performance of the model and its R2 performance is logged.
3. When model is fitted, API's endpoint over the prediction function of the model is run.
4. For all the flats that were scrapped in the first point, API request for prediction of the price is made. Afterwards difference between predicted price and actual price is calculated.
5. Email with top 15 underpriced (where difference between predicted and actual price is the biggest) flats is sent to my email address.

## How to run

1. Make folder, where data will be stored:
```bash  
mkdir data
```
2. Make folder, where model will be stored:
```bash  
mkdir models
```

3. Make folder, where whole repository will be stored:
```bash  
mkdir sreality_anomaly_detector
```

4. switch to this folder
```bash  
cd sreality_anomaly_detector
```

5. clone this repository
```bash  
git clone git@github.com:pacaklu/sreality_anomaly_detector.git
```
6. Run anomaly detection
```bash  
bash run_all.sh
```

