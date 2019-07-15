# Salary Predictions Portfolio

Problem definition: 
Examine a set of job postings with salaries and make a prediction

## 1 Explore the data
Summarize that list in a meaningful way.
Exploratory analysis showed on which fields have a greater influence on salaries and how they may be related.

## 2 Clean the data and generate new features
- Remove rows where salary=0
- Create a feature experience_bins, that assign each job posting to a group
- Create aggregation features  

## 3 Choose algorithms
I create three base models:
- Lightgbm score: 354.859
- Xgboost score: 356.649
- RandomForest score: 372.217

## 4 Tune the models
Hyperparameters tuning with Random Search

## 5 Evaluate the results
Used MSE to evaluate results. 
MSE basically measures average squared error of our predictions. For each point, it calculates square difference between the predictions and the target and then average those values.
After hyperparameters tuning MSE = 354.1 after 2115 boost rounds. 
