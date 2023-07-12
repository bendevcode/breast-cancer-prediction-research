# breast-cancer-prediction-research
# Title: Impact of Feature Selection on Model Performance in Breast Cancer Prediction using the Breast Cancer Wisconsin Dataset

Author: Oluyori Oluwagbemiga Benjamin

Affiliation: Munster Technological University, Cork, Ireland

Email: Oluwagbemiga.oluyori@cit.ie

Abstract:
Breast cancer classification using machine learning techniques has gained significant attention due to its potential for early detection and improved patient outcomes. This research paper explores the development and evaluation of machine learning models for breast cancer diagnosis prediction using the breast cancer Wisconsin dataset. The study investigates the impact of feature selection techniques on model performance and conducts hyperparameter optimization for selecting the best-performing model. Feature selection techniques, including Univariate Selection, Recursive Feature Elimination (RFE), Feature Importance, Principal Component Analysis (PCA), and L1 Regularization (Lasso), are evaluated. The models are trained and evaluated using various performance metrics, comparing models with and without feature selection. Results show that before feature selection, SVM outperforms other models in terms of Accuracy, F1 Score, and ROC AUC while after feature selection, the performance of all models decreased in most cases. Overall, it seems that feature selection doesn't improve the models' performance on this dataset, and in most cases, hyperparameter optimization did not improve the performance of the selected models across various feature selection methods. This research contributes to breast cancer classification using machine learning, emphasizing the significance of feature selection and hyperparameter optimization for accurate diagnosis prediction.

Keywords: breast cancer prediction, machine learning, feature selection, hyperparameter optimization, Wisconsin dataset

I. Introduction:
Breast cancer remains one of the most significant health threats to women worldwide. The early detection of breast cancer can greatly improve treatment outcomes, emphasizing the importance of efficient and accurate diagnostic tools. This study focuses on the Breast Cancer Wisconsin dataset, one of the most extensively used datasets in this domain. The dataset includes features extracted from digitized images of fine-needle aspirate (FNA) of breast masses. The study aims to predict whether a given sample is benign or malignant, a binary classification problem with significant real-world implications. The dataset contains 30 attributes, and various machine learning techniques will be applied to develop accurate prediction models.

II. Research:
This study investigates the role and impact of feature selection on model performance in predicting breast cancer diagnosis. Feature selection techniques, including Univariate Selection, Recursive Feature Elimination (RFE), Feature Importance, Principal Component Analysis (PCA), and L1 Regularization (Lasso), will be evaluated. Additionally, the study explores the effect of hyperparameter optimization on the performance of different machine learning models. The models considered in this study are Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and Gradient Boosting.

III. Methodology:
The research methodology involves data pre-processing, initial model building, hyperparameter optimization, and the application of feature selection techniques. The Breast Cancer Wisconsin dataset is pre-processed by handling outliers, handling missing values, transforming categorical variables, and standardizing features. Initial model building is performed using various machine learning models. Hyperparameter optimization is conducted using GridSearchCV to fine-tune the models' parameters. Feature selection techniques are applied to identify the most relevant subset of features for breast cancer prediction.

IV. Evaluation:
The evaluation involves a comprehensive analysis of the model's performance using various evaluation metrics such as accuracy, precision, recall, F1 score, and ROC AUC. The results of the initial model building phase and the optimized outcomes after feature selection and hyperparameter optimization are compared and analyzed.

V. Conclusion and Future Work:
The study concludes that feature selection techniques, in most cases, did not improve the models' performance on the Breast Cancer Wisconsin dataset. Hyperparameter optimization also did not significantly impact the accuracy of the models, except in the case of SVM after recursive feature selection. The research provides valuable insights into the efficiency and accuracy of machine learning models for breast cancer diagnosis prediction. Future work could involve exploring additional machine learning models, employing class imbalance strategies, and utilizing advanced hyperparameter tuning techniques to further improve model performance.

Keywords: breast cancer prediction, machine learning, feature selection, hyperparameter optimization, Wisconsin dataset
