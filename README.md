# ML_mastery_training

This repo contains any coding I've done using Jason Brownlee's [Machine Learning Mastery](https://machinelearningmastery.com) resources.  Descriptions of each resource are repeated here so that I have everything collected 'under one roof' so to say.

## Python Machine Learning Mini-Course

The following lessons are from the [Python ML Mini-course](https://machinelearningmastery.com/python-machine-learning-mini-course/).

### Lesson 6: Prepare for Modeling by Pre-Processing Data

Your raw data may not be setup to be in the best shape for modeling.

Sometimes you need to preprocess your data in order to best present the inherent structure of the problem in your data to the modeling algorithms.  The goal of this lesson is to use pre-processing capabilities provided by the scikit-learn package.

The scikit-learn library provides two standard idioms for transforming data.  Each transform is useful in different circumstances: Fit and Multiple Transform and Combined Fit-And-Transform.

There are many techniques that you can use to prepare your data for modelling. Some of the following are explored:

* Standardise numerical data (e.g. mean of 0 and standard deviation of 1) using the scale and center options
* Normalise numerical data (e.g. to a range of 0-1) using the range option
* Explore more advanced feature engineering such as Binarizing.

### Lesson 7: Algorithm Evaluation with Resampling Methods

The dataset used to train a machine learning algorithm is called a training dataset.  The dataset used to train an algorithm cannot be used to give you reliable estimates of the accuracy of the model on new data.  This is a big problem because the whole ideas of creating the model is to make predictions on new data.

You can use statistical methods called resampling methods to split your training dataset up into subsets, some are used to train the model and others are held back and used to estimate the accuracy of the model on unseen data.

The goal of this lesson is to practice using the different resampling methods available in scikit-learn, for example:

* Split a dataset into training and test sets
* Estimate the accuracy of an algorithm using k-fold cross validation
* Estimate the accuracy of an algorithm using leave one out cross validation

### Lesson 8: Algorithm Evaluation Metrics

There are many different metrics that you can use to evaluate the skill of a machine learning algorithm on a dataset.

You can specify the metric used for your test harness in scikit-learn via the cross_validation.cross_val_score() function and defaults can be used for regression and classification problems.

This lesson focuses on practising using different algorithm performance metrics available in the scikit-learn package.

* Practice using the Accuracy and LogLoss metrics on a classification problem
* Practice generating a confusion matrix and a classification report
* Practice using RMSE and RSquared metrics on a regression problem

