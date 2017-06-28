---
layout: post
title: Google Summer of Code - Month 1 Recap
# bigimg: /img/gsoc-logo.png
tags: [gsoc, data-science, machine-learning]
---

Over the past few weeks, I worked on a data project for patient monitoring and decision support using health data. I've summarized most of my work in the following blog post. While this current post will describe the approach I've taken for the problem, my weekly blog posts are more detailed and follow a more story-like structure. All the code and notebooks can be found in the following [link](https://github.com/olinguyen/gsoc2017-shogun-dataproject).

## Introduction

Everyday, the healthcare industry creates large amounts of patient and clinical data and stores them in electronic health records. Most of this data has previously been inaccessible, in part due to patient privacy concerns, which poses a challenge to researchers working on the analysis of health records.

However, initatives like the Medical Information Mart For Intensive Care (MIMIC) database project have allowed for everyone to use and experiment with health data. In particular, the [MIMIC database](https://mimic.physionet.org/
) is a critical care database made freely available for researchers around the world to develop and evaluate intensive care unit (ICU) patient monitoring and decision support systems that will improve the efficiency, accuracy and timeliness of clinical decision-making in critical care.

## Objective

Using the MIMIC database, I focused on these 2 prediction tasks:

1. Mortality prediction
2. Hospital length of stay  

More specifically, I'm interested in accomplishing the following:

1. Extract predictor variables from the MIMIC database
2. Build machine learning models for the predictions tasks
3. Improve performance results using hyper-parameter tuning, or ensemble methods for tree classifiers

## Overview of MIMIC

The MIMIC database mainly includes demographic, administrative, clinical data and much more from thousands of critical care patients. The table below provides basic descriptive statistics of the patients.

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

Predictors from three main categories were extracted: demographic information, vital sign data and laboratory measurements.

| Demographic & Clinical Info             | Description |
|-------------------------|---------------------------------------------|
| Age                     | Age of the patient upon entering the ICU    |
| Gender                  | Patient gender (male or female)             |
| Hospital length of stay | Number of days spent in the hospital        |
| ICU length of stay      | Number of days spent in the ICU             |
| First care unit         | ICU type in which the patient was cared for |
| Admission type          | Admission type the patient entered          |

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

To get a better grasp of the effects of the predictors on the mortality outcomes, we explored the following visualizations.

![](/img/week3/3dplot.png "3D plot")

![](/img/week3/aniongap-density.png "Anion Gap Density")

![](/img/week4/pca-2d.png "PCA 2D Plot")

![](/img/week4/t-sne.png "t-SNE")

## Preprocessing

### Exclusions

Because MIMIC is an ICU database, the focus will be on patients admitted to and discharged from the ICU. Patients admitted to the ICU generally suffer from severe and life-threatening illnesses and injuries which require constant, close monitoring and support. Being able to make good decisions during this time period is therefore crucial. For that reason, I selected and grouped the data points based off the ICU stay rather than the individual patient to develop a model specifically for ICU patient monitoring and decision-making.

The selection criteria is described below along with a short explanation. The following points were excluded from the dataset:

* Patients aged less than 16 years old
    * This also removed neonates and children, which likely have different predictors than adults
* Second admissions of patients
    * Simplifies analysis which assumes independent observations
    * We avoid taking into account that ICU stays are highly correlated
* Length of stay less than 2 days
    * Helps remove false positives that we're placed in ICU for precautionary purposes
* Exclude patients based on hospital services
    * Makes a more homogenous group of patients since we remove patients undergoing surgery

From a total of 61,534 unique ICU stay observations, the summary of the exclusions is as follows:

| Exclusion      | # ICU stays    |
|----------------|----------------|
| Length of stay | 29211 (47.47%) |
| Age            | 8109 (13.18%)  |
| First stay     | 15058 (24.47%) |
| Surgical       | 18225 (29.52%) |
| Total          | 48929 (79.52%) |

After the exclusions, up to 48929 observations (almost 80% of samples), which is quite significant. The remaining data consists of a little over 20,000 patients that are kept for data analysis.

### Data cleaning

This introduced the issues of much more NaN values being present in the data because not all lab measurements are recorded for every patient. To circumvent this, we will make use of the `pandas` library to deal with missing data. More specifically, the data imputation technique, or the method of replacement of the missing data, will employ mean substitution which will replace missing values with the mean value of that feature. Doing so allows us to increase our dataset size by 3,000, with a total of over 32,000 data points. Additionally, we shall add [data normalization](https://en.wikipedia.org/wiki/Feature_scaling ) to preprocess the data. Because our data has very different features that have different metrics, units and scales, we will standardize the data by making each feature have zero mean by subtracting the mean, and have unit-variance. Feature scaling ensures that all the data is normalized, that the features are in the same range. Some algorithms like the [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) can converge faster on normalized data.

## Model & Training

For mortality prediction, we mainly used two machine learning classifiers: logistic regression and linear support vector machine. These algorithms are commonly used and allow learn the relationship between predictor variables and a binary outcome variable.

## Results

| Classifier          | Mean AUC across 10 folds (%) |                  |                    |               |
|---------------------|--------------------------|------------------|--------------------|---------------|
|                     | 1-year mortality         | 30-day mortality | Hospital mortality | ICU mortality |
| Logistic Regression | 78.61                    | 82.05            | 84.64              | 85.35         |
| Linear SVM          | 78.57                    | 82.17            | 84.56              | 85.18         |
| Random guess         |          0.50           |          0.50   | 0.50             |      0.50    |
| Logistic Regression (sklearn)         |                     |             | 84.64              |          |
| Linear SVM (sklearn)         |                     |             | 84.56              |          |
| XGBoost         |                     |             | 87.60              |          |




| Classifier              | Mean MSE across 10 folds |                    |
|-------------------------|--------------------------|--------------------|
|                         | Hospital length of stay  | ICU length of stay |
| Least square regression | 110.726                  | 34.878             |
| Linear ridge regression | 110.726                  | 34.878

![](/img/week3/mp-results.png "Mortality prediction results")

![](/img/week3/boxplot-mp.png "Boxplot mortality prediction")

![](/img/week3/roc-curve.png "ROC Curve")

### Comparisons with other frameworks

The training time for logistic regression and linear SVM were compared between sklearn and shogun.

| Classifier | Mean train time across 10 folds (seconds)  |
|--|
|                     | shogun | sklearn |
| Linear SVM          | 4.516       |   12.05 |
| Logistic Regression | 6.265       |   0.7488  |

## Next Up
