# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model was created by Triet Le for ML DepOps Nanodegree from Udacity
- Date 23/10/2022
- Version 1.0
- Random Forest Classifier of scikit-learn was used
- For starter code, license and liability, please contact Udacity Nanodegree program

## Intended Use
- This is an example project for Deploying a Machine Learning Model in Production
- Predict whether income exceeds $50K/yr based on census data

## Training Data
- Extraction was done by Barry Becker from the 1994 Census database. 
- A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) 

## Evaluation Data
- 20% of the original dataset was used for evaluation purposes of the model.

## Metrics
- Metrics fbeta_score, precision_score and recall_score were used in this project.

## Ethical Considerations
- Not included 

## Caveats and Recommendations
- Model results can be improved by tunning the hyperparameters and more advanced classifiers.
