# creditDefaults
Consumer credit default analysis

## Overview 

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this is often the most marginalised part of the population. The approach taken in this analysis takes a naive approach to understading consumer defaults due to the rather limited nature of the data. The data set used in this report includes consumer default instances, defined in the data as the TARGET variable. A default is defined to be the customer that has late payment more than 90 days on a given load or failed to repay the loan.

The borrower characteristics include quantative metrics such as income, credit amount, value of good purchased, days employed etc. It also include qualatative/catergorical meausures such as gender, education, home ownership, mobile ownership, number of children etc.

This analysis does not take into account financial inclusion for the unbanked population. In fronteir markets, its important to make sure that underserved populations has a positive loan experience and given everyone an fair assessment. This data does not take into account alternative data sources to predict customers repayment abilities.

In this report an array of classifiers are explored. Each model is fitted to the data using a training sample. The testing sample is used to evaluate each model's performance in predicting credit defaults. The model fits can be seen in the Annex. Once the models have been fitted using the training data, each model's performance can be evaluated when applied to unseen data (out-of-sample).The algorithms include logistic, K-nearest neighbours, random forest, decision trees, ada boost, naive Bayes and gradient boosted classifier.
