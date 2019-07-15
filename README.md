# Salary Predictions Portfolio

Problem definition: 
Examine a set of job postings with salaries and make a prediction.
The goal is to predict the salary of a job postings based on the given information.

## 1. Explore the data
Summarize the data in a meaningful way.
- Identify patterns and outliers.
- Examine the distribution of the variable and relationship between features. 
- Exploratory analysis of features and their influence on salaries.

## 2. Clean the data and generate new features
- Remove rows where salary=0
- Create a feature experience_bins, that assign each job posting to a group
- Create aggregation features  

## 3. Choose algorithms
I created three base models:
- Lightgbm score: 354.859
- Xgboost score: 356.649
- RandomForest score: 372.217

## 4. Tune the models
As LightGBM showed better results I will try to improve score by hyperparameters tuning with Random Search

## 5. Evaluate the results
I used MSE to evaluate results. 
MSE basically measures average squared error of our predictions. For each point, it calculates square difference between the predictions and the target and then average those values.
After hyperparameters tuning the results was a bit improved MSE = 354.1 after 2115 boost rounds. 

