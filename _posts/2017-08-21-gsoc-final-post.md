---
layout: post
title: Google Summer of Code - Final Blog Post
# bigimg: /img/gsoc-logo.png
tags: [gsoc, data-science, machine-learning]
---

The following blog post summarizes most of my work for the [Google Summer of Code 2017](https://developers.google.com/open-source/gsoc) with the [Shogun Machine Learning Toolbox](shogun.ml). A python notebook that accompanies this blog post can be found [here](https://github.com/olinguyen/gsoc2017-shogun-dataproject/blob/master/Shogun%20Showroom.ipynb).

## Introduction

Everyday, the healthcare industry creates large amounts of patient and clinical data and stores them in electronic health records. Most of this data has previously been inaccessible, in part due to patient privacy concerns, which poses a challenge to researchers working on the analysis of health records.

However, initatives like the Medical Information Mart For Intensive Care (MIMIC) database project have allowed for everyone to use and experiment with health data. In particular, the [MIMIC database](https://mimic.physionet.org/
) is a critical care database made freely available for researchers around the world to develop and evaluate intensive care unit (ICU) patient monitoring and decision support systems that will improve the efficiency, accuracy and timeliness of clinical decision-making in critical care.

## Table of Contents
1. [Objective](#1-objective)
2. [Overview of MIMIC](#2-overview-of-mimic)
    2.1 [Features](#21-features)
    2.2 [Visualization](#22-visualization)
3. [Preprocessing](#3-preprocessing)
    3.1. [Exclusions](#31-exclusions)
    3.2. [Data cleaning](#32-data-cleaning)
4. [Basic model](#4-basic-model)
    4.1. [Results](#41-results)
5. [Improved model with temporal features](#5-improved-model-with-temporal-features)
    5.1 [Resampling](#51-resampling)
    5.2 [Model](#52-model)
    5.3 [Results](#53-results)
8. [Conclusion](#conclusion)
9. [Future improvements](#future-improvements)

## 1. Objective

Using the MIMIC database which includes vital signs, medications, diagnostic code and many more variables, along with feature engineering techniques, I investigated whether I could build a robust classifier to perform the following tasks:

1. Mortality prediction
2. Hospital length of stay  

I first experimented with a basic model which ignores temporal structure in the data, and attempted to perform mortality prediction. Briefly, for the first model, I completed the following tasks:

1. Extracted predictor variables from the MIMIC database
2. Built machine learning models for the predictions tasks
3. Evaluated and compared the performance of various algorithms 

In the second improved model, I took into account the complexity of time series data, since past vital signs and how they previously fluctuated can be strong indicators of the current condition of a patient. The problem is still modeled as a binary classification problem, but now with time series data as a predictor. In this part, I had to compute features to model temporal data and use it for mortality prediction at any point in time.

From there, I explored the pros and cons of each model, the challenges of using time series data and compared the different approaches while providing discussions on the analysis & results.

## 2. Overview of MIMIC

The MIMIC database mainly includes demographic, administrative, clinical data and much more from thousands of critical care patients. The table and the plot below provides basic descriptive statistics of the patients and an overview of the dataset.

| Information                    | Totals |
|--------------------------------|-------------|
| Age, years, median          data-cleaning   | 65.769      |
| Gender, male (%)               | 56.207      |
| Distinct number of patients    | 38,597      |
| Distinct ICU stays             | 53,423      |
| Hospital admissions            | 49,785      |
| Hospital length of stay (days) | 11.545      |
| ICU length of stay (days)      | 2.144       |
| Hospital mortality (%)         | 11.545      |
| ICU mortality (%)              | 8.545       |

![](/img/week2/hist-mimic.png "Histograms for MIMIC")

Additional information regarding the MIMIC database can be found in the [published paper](https://www.nature.com/articles/sdata201635) as well.

### 2.1 Features

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

I first experimented with a basic model which ignores temporal structure in the data, and attempted to perform mortality prediction. In the second improved model, I incorporated lagged features which takes into account the complexity of time series data. From there, I explored the pros and cons of each model, the challenges of using time series data and discussions on the analysis & results.

## 2. Overview of MIMIC

The MIMIC database mainly includes demographic, administrative, clinical data and much more from thousands of critical care patients. The table and the plot below provides basic descriptive statistics of the patients and an overview of the dataset.

| Information                    | Totals |
|--------------------------------|-------------|
| Age, years, median          data-cleaning   | 65.769      |
| Gender, male (%)               | 56.207      |
| Distinct number of patients    | 38,597      |
| Distinct ICU stays             | 53,423      |
| Hospital admissions            | 49,785      |
| Hospital length of stay (days) | 11.545      |
| ICU length of stay (days)      | 2.144       |
| Hospital mortality (%)         | 11.545      |
| ICU mortality (%)              | 8.545       |

![](/img/week2/hist-mimic.png "Histograms for MIMIC")

Additional information regarding the MIMIC database can be found in the [published paper](https://www.nature.com/articles/sdata201635) as well.

### 2.1 Features

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

During a patients' stay in the ICU, their routine vital signs and additional information relevant to their care are constantly monitored and recorded electronically. This results in data taking the form of a time series where there is an ordered sequence of observations of many variables. I extracted every vital sign data point of the first 24 hours of a patient upon entering the ICU. The raw values of a few vital signs of a patient looks as follows:

![](/img/week10/single-patient-vitals.png "Patient Vital Signs")

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

Laboratory measurements taken from a patient are also strong indictators of a patient's health condition. Let's take anion gap as an example. [Anion gap](https://en.wikipedia.org/wiki/Anion_gap) is the difference between primary measured cations (sodium Na+ and potassium K+) and the primary measured anions (chloride Cl- and bicarbonate HCO3-) in [serum](https://en.wikipedia.org/wiki/Serum_(blood)) (blood). The test is mostly performed in patients with altered mental status, unknown exposures, acute renal failure, and acute illnesses [1]. A kernel density estimation plot is used to view the distribution of the values below shows the aniongap measurement on ICU admission comparison for survival and non-survival groups.

![](/img/week3/aniongap-density.png "Anion Gap Density")

### 2.2 Visualization

To get a better grasp of the effects of the predictors on the mortality outcomes, I explored the dataset with multiple visualizations.

Here, I used a 3d plot to see how different features affect a patient's mortality probability.

In this visualization, I included age, heart rate and the [Glasgow Coma Scale](https://en.wikipedia.org/wiki/Glasgow_Coma_Scale) which is a score indicating the level of consciousness of a person. From the plot, patients with somewhat extreme values (low heart rate and Glasgow Coma Scale values) are more likely to die, shown with a red 'X'. Although this plot only shows 3 predictors, it is possible to change the variables on the 3 axis for visualizations.

![](/img/week3/3dplot.png "3D plot")

In total, 48 features are used to build the model for both prediction tasks. Since there are a high number of features, I was interested in seeing how high-dimensional data would look like in 2D using dimensionality reduction techniques like PCA and t-SNE.

![](/img/week4/pca-2d.png "PCA 2D Plot")

![](/img/week4/t-sne.png "t-SNE")

From the plots, it can be noticed that there is a noticeable difference with each group forming its own cluster.

## 3. Preprocessing

### 3.1. Exclusions

Because MIMIC is an ICU database, the focus was placed on patients admitted to and discharged from the ICU. Patients admitted to the ICU generally suffer from severe and life-threatening illnesses and injuries which require constant, close monitoring and support. Being able to make good decisions during this time period is therefore crucial. For that reason, data points were queried and grouped based off the ICU stay rather than the individual patient to develop a model specifically for ICU patient monitoring and decision-making.

The selection criteria is described below along with a short explanation. The following points were excluded from the dataset:

* Patients aged less than 16 years old
    * This also removed neonates and children, which likely have different predictors than adults
* Second admissions of patients
    * Simplifies analysis which assumes independent observations
    * We avoid taking into account that ICU stays are highly correlated
* Length of stay less than 1 day
    * Helps remove false positives that were placed in ICU for precautionary purposes

### 3.2 Data cleaning

Because not all lab measurements are recorded for every patient, a lot missing values and NaNs were found in the dataset which were replaced with the mean value.

 Additionally, data standardization was applied to make each feature have zero mean by subtracting the mean, and have unit-variance to ensure that all the data is normalized, that the features are in the same range.

## 4. Basic model

In this first model, I investigated how well predictions could be when not using time series data. I ignored the temporal structure of the data, and only used the entire data of the first 24 hours in the ICU for a patient. A single time-shot computing the minimum, maximum and mean for that time period was used for the vital signs. For mortality prediction, three machine learning classifiers were used: random forest, logistic regression and linear support vector machine. These algorithms are commonly used and allow to learn the relationship between predictor variables and a binary outcome variable.

### 4.1 Results

Using [stratified 10-fold cross-validation]([1]), the [auROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) was the metric used to evaluate the performance of the classifiers for mortality prediction and recorded in the table below. The scores were also compared with sklearn's implementation of logistic regression and linear SVM and yielded identical results. Finally, I compared the result with [XGBoost](xgboost.readthedocs.io), which is a popular algorithm used in Kaggle competitions.

| Classifier          | Mean AUC across 10 folds (%) |
|---------------------|--------------------|
|                     | Hospital mortality |
| Logistic Regression | 84.64              |
| Linear SVM          | 84.56              |
| Logistic Regression (sklearn)         |  84.64              |
| Linear SVM (sklearn)         |    84.56              |    
| XGBoost             | 87.60              |  
| Random guess         |          50.00           |        

For all three classifiers, 1-day and 1-week mortality predictions were evaluated. The barcharts below show the results for the different tasks.

![](/img/finalweek/mp-results.png "Mortality prediction results")

Boxplots give an indication of the variance of the results over the 10 folds through cross-validation.

![](/img/finalweek/basic-boxplot-7day.png "Boxplot mortality prediction")

 The [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for hospital mortality gives an insight on the sensitivity and specificity of our logistic regression and linear SVM models. The performance of both models were quite similar.

![](/img/week3/roc-curve.png "ROC Curve")

Finally, the regression task for predicting hospital length of stay was evaluate using mean squared error.

| Classifier              | Mean MSE across 10 folds |                      |          
|-------------------------|--------------------------|----------------------|
|                         | Hospital length of stay  | ICU length of stay   |  
| Least square regression | 110.726                  | 46.91                |
| Linear ridge regression | 110.726                  | 46.91                |

[1]: (https://en.wikipedia.org/wiki/Cross-validation_\(statistics›)#k-fold_cross-validation)

------

## 5. Improved model with temporal features

The previous basic model above that was built for mortality prediction was based on a single time-shot features i.e. vital signs of the first 24 hours were aggregated and had descriptive statistics computed. As a result, the basic model fails to account any temporal structure in the data and will not capture a changing health status of a patient.

Let's recall that the objective was to develop a model that could allow caregivers to monitor and predict mortality of patients. In practice, such a system would be incorporated in a real-time clinical monitoring system where it would be possible to look at a patient at any point in time and make a prediction about whether the patient will die within a certain amount of time. The previous model is limited in the sense that it doesn't account for predictions at any point in time, since it's trained with data from only the first 24 hours combined.

In this improved model, I exploit the temporal information, taking into account the fluctuations of the vital signs over time and most importantly does not look at how the vital signs change moments before death. From there, it allowed me to better understand how a dying patient differs from a patient with normal behavior moments before days, hours and right before death.

### 5.1 Resampling

One of the challenges with the MIMIC database is that vital signs are measured and recorded at irregular time intervals. Resampling refers to converting raw time series data into discrete intervals at a fixed frequency. A smaller time resolution between vital sign readings, the better and more accurate the classifier becomes since it allows for timely predictions. 

Because space and computing power is limited, a sampling rate measuring capturing vital signs every hour was chosen, for the first 24 hours of a patient. Afterwards, lagged features from the last 3 hourly observations (t-1, t-2, t-3), which are the vital signs at the previous hours, were computed by shifting columns of the time stamp. 

A rolling window then computes the mean, min, max and median over the entire vital signs time series of a patient in the last 6 hours.  The same vital signs used previously were included for this model. Below is a sample of data points with only heart rate used.

| timestamp           | hr | hr_1h | hr_max_6h | hr_mean_6h | hr_median_6h | hr_min_6h |
|---------------------|---------------|-------|-----------|------------|--------------|-----------|
|  13:00 | 90.0          | 74.0  | 90.0      | 79.85      | 77.0         | 74.0      |
|  14:00 | 81.0          | 90.0  | 90.0      | 79.28      | 77.0         | 74.0      |
|  15:00 | 79.0          | 81.0  | 90.0      | 79.00      | 77.0         | 74.0      |
|  16:00 | 79.0          | 79.0  | 90.0      | 79.42      | 79.0         | 74.0      |
|  17:00 | 68.0          | 79.0  | 90.0      | 78.14      | 79.0         | 68.0      |

To observe how these features differ between dying and surviving patients, I plotted stacked histograms from equally sampled survival and non-survival patients to view the distribution of these vital signs values across both groups.

![](/img/finalweek/vitals-comparison-1day.png "Vital Sign comparison")

While there is no clear conclusion that can be drawn from looking at the vital signs histograms, this may indicate that these variables vary depending on the patient. For instance, a patients'

### 5.2 Model

In the end, the features included vital signs, their lagged observations and window computations, demographic information and the hospital length of stay of the patient. The feature matrix consisted of a single row represented a patient observation at a certain point in time with the features mentionned above. The following plot shows what sample features look like for a dying patient.

![](/img/week10/features-patient.png "Patient Features")

The label for that feature vector was whether that patient died within 1 day or within 1 week of time, which results in a large feature matrix consisting rows of patient data over the past 6 hours at different points in time, and whether the patient will die in 1 day and 1 week.

### 5.3 Results

Three classifiers (logistic regression, linear SVM and random forest) were evaluated and compared. One challenge I faced was dealing with class imbalance where the number of negative samples (patients that did not die) was significantly larger than the number of positive samples (patients that died). This was addressed by undersampling the negative samples to maintain a ratio of 80/20. In addition, because the size of the dataset consisted of over 500,000 data points, the training and testing set were subsampled and evaluated several times. The subset of the data consisted of 100,000 points and split in two for training and testing, and averaged over 10 runs. The results were recorded in the following table.

| Classifier          | Test set AUC score (%)    |                            |
|---------------------|---------------------------|----------------------------|
|                     | 1-day hospital mortality  | 1-week hospital mortality   |
| Logistic Regression | 73.48                     | 72.03                      |
| Linear SVM          | 73.22                     | 71.81                      |
| Random Forest       | 83.12                     | 83.51                      |


![](/img/finalweek/boxplot-1day.png "1 Day Boxplot")

Random forest produced the best results among all three classifiers for 1-day and 1-week mortality prediction. The variance across the 10 runs with 100,000 samples from the entire dataset can also be seen here.

![](/img/finalweek/boxplot-1week.png "1 Week Boxplot")

The scores for 1-day mortality were slightly higher than 1-week mortality, which could be explained due to the fact that the first 24 hours upon entering the ICU are critical and are closer in time to the prediction period of 1 day.

## Conclusion

I first started off with a basic model using demographic information, vital signs and laboratory measurements for predicting patient mortality. Using three common machine learning classifiers, I was able to obtain decent results of 85% AUC. However, ignoring the temporal component of the data reduced the predictive power of my models, since I used a single time-shot of all vital signs to predict mortality. In the second,improved model, I added lagged features and computed sliding window summary statistics for the vital signs. The latter model ended up being much more realistic and capable of making better predictions at any point in time since it took into account the time series data.

## Future improvements

While I used lagged features and sliding window summary statistics for the time series data, there exists many other methods of modeling data of time. Recurrent neural networks have shown to be effective, especially with sequences and temporal data.

Because of the size and complexity of the MIMIC database, there are many other types of problems that can tackled. Here are some challenging areas that were beyond the scope of this data project, but that would have been interesting to include.

* Patient similarity
* Survival analysis (time-to-event prediction)
* Disease prediction
