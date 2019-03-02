# Predict Doctor's Consultation Fee

[hackathon info](https://www.machinehack.com/course/predict-a-doctors-consultation-fees-hackathon/)

[view solution notebook](https://nbviewer.jupyter.org/github/NishantBhavsar/doctor-consultation-fee-prediction/blob/master/code/Analysis%20%26%20Data%20Preparation.ipynb)

### Problem Statement
We have all been in situation where we go to a doctor in emergency and find that the consultation fees are too high. As a data scientist we all should do better. What if you have data that records important details about a doctor and you get to build a model to predict the doctor’s consulting fee.?


### Libraries
```
scikit-learn==0.19.2
pandas==0.23.4
matplotlib==2.2.2
seaborn==0.9.0
catboost==0.9.1.1
```

### Data Features
1. `Qualification`: Qualification and degrees held by the doctor
2. `Experience`: Experience of the doctor in number of years
3. `Rating`: Rating given by patients
4. `Profile`: Type of the doctor
5. `Miscellaeous_Info`: Extra information about the doctor
6. `Place`: Area and the city where the doctor is located.
7. `Fees`: Fees charged by the doctor

### Data Pre-processing and Feature creation
1. `Qualification` columns has all the study records of a doctor in a string format by comma separated.
E.g. MBBS, MS - Otorhinolaryngology. So, it makes sense to create features out of it like Diploma, Bachelor, Masters, and Extra study, etc., but the problem is we have more than 600 unique study and major combination and we need understanding of all these Medical course to created different features, that's why I have TfIdf vector and TurncatedSVD on these features to create final 20 features out of this Qualification information.

2. Extracted just number out of `Experience`.

3. Removed % from `Rating`.

4. More than 3000 records doesn't have `Miscellaeous_Info` and in majority case it only consists info about Experience and location, which we already have, so I am not using this column in model. I have created an extra column `Has_M_Info` which indicates if the record has miscellaeous info or not.

5. `Place` column has Area and City info comma separated, so I have created two columns `Area` & `City` out of it.

### Final Fetures for model creation
1. 20 features from `Qualification`
2. `Experience`
3. `Rating`
4. `Profile`
4. `Has_M_Info`
6. `City`
- I haven't used `Area`, because it has 800+ unique values, which can overfit our model.

### EDA
- After doing data analysis it is clear that Rating and Experience has less correlation with target variable Fees.
- It was clear from Box plots of Fees for Profile and City that these columns are more important for segregation.
- But without adding Education information we can still not find good separation in data groups.

### Evaluation metric
- Root mean squared log error (RMSLE)
- The reason so select this metric is because out target variable is skewd, so we don't want to add more penalty for extream values in out machine learning model.

### Model
- I have used Catboost Regressor model
- Mean squared error (MSE) is used as a loss function and RMSLE as a evalution metric.
- `Profile` and `City` is used as categorical variables.
- Used 5 Fold __Cross validation__ and averaged predictions from each fold for test dataset for final submission.

### Final model result 
- Achieved Validation RMSLE of 0.625 and RMSE of 168.87

### How to further improve the result?
- If we created more features out of `Experience` column by understanding of all medical courses, then there is a high chance to improve our model performance.
- Try different model like Xgboost, LinearRegression, etc. and do ensembling.
