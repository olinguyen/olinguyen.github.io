---
layout: post
title: Google Summer of Code - Month 1 Recap
# bigimg: /img/gsoc-logo.png
tags: [gsoc, data-science, machine-learning]
---

Over the past few weeks, I worked on a data project for patient monitoring and decision support using health data. I've summarized most of my work in the following blog post. While this current post will describe the approaches that I've taken for the problem, my weekly blog posts are more detailed and read more as story. All the code and notebooks can be found in the following [link](https://github.com/olinguyen/gsoc2017-shogun-dataproject).

## Introduction

Everyday, the healthcare industry creates large amounts of patient and clinical data and stores them in electronic health records. Most of this data has previously been inaccessible, in part due to patient privacy concerns, which poses a challenge to researchers working on the analysis of health records.

However, initatives like the Medical Information Mart For Intensive Care (MIMIC) database project have allowed for everyone to use and experiment with health data. In particular, the [MIMIC database](https://mimic.physionet.org/
) is a critical care database made freely available for researchers around the world to develop and evaluate intensive care unit (ICU) patient monitoring and decision support systems that will improve the efficiency, accuracy and timeliness of clinical decision-making in critical care.

## Objective

Using the MIMIC database, I focused on these 2 prediction tasks:

1. Mortality prediction
2. Hospital length of stay  

More specifically, I accomplished the following:

1. Extracted predictor variables from the MIMIC database
2. Built machine learning models for the predictions tasks
3. Evaluated and compared the performance of various algorithms

## Overview of MIMIC

The MIMIC database mainly includes demographic, administrative, clinical data and much more from thousands of critical care patients. The table and the plot below provides basic descriptive statistics of the patients and an overview of the dataset.

| Information                    | Totals |
|--------------------------------|-------------|
| Age, years, median             | 65.769      |
| Gender, male (%)               | 56.207      |
| Distinct number of patients    | 38,597      |
| Distinct ICU stays             | 53,423      |
| Hospital admissions            | 49,785      |
| Hospital length of stay (days) | 11.545      |
| ICU length of stay (days)      | 2.144       |
| Hospital mortality (%)         | 11.545      |
| ICU mortality (%)              | 8.545       |

![](/img/week2/hist-mimic.png "Histograms for MIMIC")

### Features/Predictors

Predictors from three main categories were extracted: demographic information, vital sign data and laboratory measurements. These were selected as the most relevant information in determining the likelihood of mortality and hospital length of stay.

| Demographic & Clinical Info             | Description |
|-------------------------|---------------------------------------------|
| Age                     | Age of the patient upon entering the ICU    |
| Gender                  | Patient gender (male or female)             |
| Hospital length of stay | Number of days spent in the hospital        |
| ICU length of stay      | Number of days spent in the ICU             |
| First care unit         | ICU type in which the patient was cared for |
| Admission type          | Admission type the patient entered          |

Vital signs are clinical measurements that describe the state of a patient's body functions.

| Vital sign               | Description |
|--------------------------|-------------|
| Heart Rate               | Heartbeat rate of the patient |
| Mean Blood Pressure      | Average pressure in a patient's arteries during one cardiac cycle       |
| Diastolic blood pressure | Pressure when the heart is at rest between beats            |
| Systolic blood pressure  | Pressure when the heart is beating |
| Respiratory Rate         | Number of breaths taken per minute       |
| Temperature              | Temperature of a patient in degrees Celcius            |
| SpO2                     | Amount of oxygen in the blood            |
| Glasgow Coma Scale       | Scoring system used to describe the level of consciousness in a person            |
| Ventilation              | Whether the patient was ventilated or not            |
| Urine output             | How much urine was produced            |

Laboratory measurements are made by acquiring a fluid from the patient's body (e.g. blood from an arterial line or urine from a [catheter](https://en.wikipedia.org/wiki/Catheter)) and then analyzing it in the laboratory.

| Laboratory measurements |
|-------------------------|
| Aniongap                |
| Bicarbonate             |
| Creatinine              |
| Chloride                |
| Glucose                 |
| Hematocrit              |
| Hemoglobin              |
| Platelet                |
| Potassium               |
| Sodium                  |
| Blood urea nitrogen     |
| White blood cells       |

### Visualization

To get a better grasp of the effects of the predictors on the mortality outcomes, I explored the dataset with multiple visualizations.

I used a 3d plot to see how different features affect a patient's mortality probability.

In this first visualization, I included age, heart rate and the [Glasgow Coma Scale](https://en.wikipedia.org/wiki/Glasgow_Coma_Scale) which is a score indicating the level of consciousness of a person. From the plot, patients with somewhat extreme values (low heart rate and Glasgow Coma Scale values) are more likely to die, shown with a red 'X'. Although this plot only shows 3 predictors, it is possible to change the variables on the 3 axis for visualizations.

![](/img/week3/3dplot.png "3D plot")

Laboratory measurements taken from a patient are also strong indictators of a patient's health condition. Let's take anion gap as an example. [Anion gap](https://en.wikipedia.org/wiki/Anion_gap) is the difference between primary measured cations (sodium Na+ and potassium K+) and the primary measured anions (chloride Cl- and bicarbonate HCO3-) in [serum](https://en.wikipedia.org/wiki/Serum_(blood)) (blood). The test is mostly performed in patients with altered mental status, unknown exposures, acute renal failure, and acute illnesses [1]. A kernel density estimation plot is used to view the distribution of the values below shows the aniongap measurement on ICU admission comparison for survival and non-survival groups.

![](/img/week3/aniongap-density.png "Anion Gap Density")

In total, 48 features are used to build the model for both prediction tasks. To visualize high-dimensional data, I employed PCA and t-SNE as dimensionality reduction techniques.

![](/img/week4/pca-2d.png "PCA 2D Plot")

![](/img/week4/t-sne.png "t-SNE")

## Preprocessing

### Exclusions

Because MIMIC is an ICU database, the focus was placed on patients admitted to and discharged from the ICU. Patients admitted to the ICU generally suffer from severe and life-threatening illnesses and injuries which require constant, close monitoring and support. Being able to make good decisions during this time period is therefore crucial. For that reason, data points were queried and grouped based off the ICU stay rather than the individual patient to develop a model specifically for ICU patient monitoring and decision-making.

The selection criteria is described below along with a short explanation. The following points were excluded from the dataset:

* Patients aged less than 16 years old
    * This also removed neonates and children, which likely have different predictors than adults
* Second admissions of patients
    * Simplifies analysis which assumes independent observations
    * We avoid taking into account that ICU stays are highly correlated
* Length of stay less than 2 days
    * Helps remove false positives that were placed in ICU for precautionary purposes

### Data cleaning

Because not all lab measurements are recorded for every patient, a lot missing values and NaNs were found in the dataset which were replaced with the mean value.

 Additionally, data standardization was applied to make each feature have zero mean by subtracting the mean, and have unit-variance to ensure that all the data is normalized, that the features are in the same range.

## Model & Training

For mortality prediction, two machine learning classifiers were used: logistic regression and linear support vector machine. These algorithms are commonly used and allow to learn the relationship between predictor variables and a binary outcome variable.

## Results

Using [stratified 10-fold cross-validation]([1]), the [auROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) was the metric used to evaluate the performance of the classifiers for mortality prediction and recorded in the table below. The scores were also compared with sklearn's implementation of logistic regression and linear SVM and yielded identical results. Finally, I compared the result with [XGBoost](xgboost.readthedocs.io), which is a popular algorithm used in Kaggle competitions.

| Classifier          | Mean AUC across 10 folds (%) |
|---------------------|--------------------|
|                     | Hospital mortality |
| Logistic Regression | 84.64              |
| Linear SVM          | 84.56              |
| Random guess         |          0.50           |        
| Logistic Regression (sklearn)         |  84.64              |
| Linear SVM (sklearn)         |    84.56              |    
| XGBoost             | 87.60              |  

In addition to mortality prediction, 30-day, 1-year and ICU mortality prediction were evaluated. The barcharts below show the results for the different tasks.

![](/img/week3/mp-results.png "Mortality prediction results")

Boxplots give an indication of the variance of the results over the 10 folds through cross-validation.

![](/img/week3/boxplot-mp.png "Boxplot mortality prediction")

 The [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for hospital mortality gives an insight on the sensitivity and specificity of our logistic regression and linear SVM models. The performance of both models were quite similar.

![](/img/week3/roc-curve.png "ROC Curve")

Finally, the regression task for predicting hospital length of stay was evaluate using mean squared error.

| Classifier              | Mean MSE across 10 folds |                  
|-------------------------|--------------------------|
|                         | Hospital length of stay  |
| Least square regression | 110.726                  |
| Linear ridge regression | 110.726                  |

The training time for logistic regression and linear SVM were compared between sklearn and shogun. While shogun has a faster training time for linear SVM when compared to sklearn, the opposite scenario occurs for logistic regression.

| Classifier | Mean train time across 10 folds (seconds)  |
|--|
|                     | shogun | sklearn |
| Linear SVM          | 4.516       |   12.05 |
| Logistic Regression | 6.265       |   0.7488  |

[1]: (https://en.wikipedia.org/wiki/Cross-validation_\(statistics›)#k-fold_cross-validation)
