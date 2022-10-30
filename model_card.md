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
-Metrics fbeta_score, precision_score and recall_score were used in this project.
-The model's performance is:
 Precision score = 0.7297904191616766
 Recall score = 0.6210191082802548
 fbeta score = 0.6710254645560908

## Ethical Considerations
- Data: Data was anonymized.
- Human life: Helpful to policy makers to improve society fairness.
- Mitigations: Limit the user range to society researchers.
- Risks and harms: Model users may derive on the discrimination issues in society.
- Use cases: Use case in other years should be followed.

## Caveats and Recommendations
- Model results can be improved by tunning the hyperparameters and more advanced classifiers.
