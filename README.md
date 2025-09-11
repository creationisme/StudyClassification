This [dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data) is related to red variants of the Portuguese "Vinho Verde" wine. The dataset describes the amount of various chemicals present in wine and their effect on its quality. The data can be used for classification tasks (predicting quality of the wine). It can also be used for regression tasks such as predicting the alcohol level in a given wine. 

This project focuses on various **classification** techniques. I have first performed extensive Exploratory Data Analysis (EDA) and necessary preprocessing techniques. The dataset is used to compare and contrast various ML classification algorithms: Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM, KNN. I have also trained a neural networks for this classification, to set a baseline. For every ML algorithm, I have run Grid Search to zero in on the best hyperparameters and then compared accuracy of each model. 

Findings:

The Neural Network performs quite poorly, reaching highest training accuracy of 66%, validation accuracy of 57% and accuracy on the test set as 53%, among multiple runs. This performance likely due to the small dataset, reflecting on a problem with deep networks, which are very data hungry. For small datasets, simple ML models seem to perform much better.

The Decision Tree Classifier, ill equipped to handle such heavy imbalance, performed quite poorly. Random Forest and XGBClassifiers, both reached high training accuracies (~91%) but were stuck at test accuracies of 63.73% and 56.86% each, indicating some form of overfitting to the training dataset. 

All three models consistently ranked alcohol levels followed by sulphate levels as the most important features for classification. Upon closer look at the pairplot done during EDA, it does seem like the distributions for alcohol levels for different classes were quite distinguishable from each other while other features seemed to have much larger overlap.

For the other classifiers, the model with the highest test accuracy was the Logistic Regression trained on the 'Without PCA' dataset, achieving an accuracy of 0.6127. It also suggested that alcohol level is the most significant positive predictor of wine quality while volatile acidity is the most significant negative predictor. Sulphates also have a notable positive impact. This agrees with our random forest and XGB models. 

All models performed similarly to or better than the neural network.

Overall, the Random Forest Classifier performed the best.
