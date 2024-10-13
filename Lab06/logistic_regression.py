import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

""" import os 
for dirname, _, filenames in os.walk('./kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) """

""" import warnings
warnings.filterwarnings('ignore')  """      

data = './kaggle/weatherAUS.csv'

df = pd.read_csv(data)

# ---Exploratory data analysis---

# print(df.shape)

# print(df.head())

col_names = df.columns

# print(col_names)

# df.drop(['RISK_MM'], axis=1, inplace=True)

# print(df.info())

# categorical = [var for var in df.columns if df[var].dtype=='O']

# print('There are {} categorical variables\n'.format(len(categorical)))

# print('The categorical variables are :\n\n', categorical)

# df[categorical].head()

# ---Explore problems within categorial variables---

# print(df[categorical].isnull().sum())

# cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

# print(df[cat1].isnull().sum())

""" for var in categorical:
    # print(df[var].value_counts()/np.float64(len(df)))
    print(var,' contains ', len(df[var].unique()), ' labels') """

# print(df['Date'].dtypes)

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
# print(df['Year'].head())

df['Month'] = df['Date'].dt.month
# print(df['Month'].head())

df['Day'] = df['Date'].dt.day
# print(df['Day'].head())

df.drop('Date', axis=1, inplace=True)

# print(df.head())

# categorical = [var for var in df.columns if df[var].dtype=='O']

# print('There are {} categorical variables\n'.format(len(categorical)))

# print('The categorical variables are :\n\n', categorical)

# print(df[categorical].isnull().sum())

# print('Location contains ', len(df['Location'].unique()), ' labels')

# print(df.Location.unique())

# print(df.Location.value_counts())

# print(pd.get_dummies(df['Location'], drop_first=True).head())

# print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

# print(df['WindGustDir'].unique())

# print(df['WindGustDir'].value_counts())

# print(pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0))

# print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')

# print(df['WindDir9am'].unique())

# print(df['WindDir9am'].value_counts())

# print(pd.get_dummies(df['WindDir9am'], drop_first=True).head())

# print(pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0))

# print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')

# print(df['WindDir3pm'].unique())

# print(df['WindDir3pm'].value_counts())

# print(pd.get_dummies(df['WindDir3pm'], drop_first=True).head())

# print(pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0))

# print('RainToday contains', len(df['RainToday'].unique()), 'labels')

# print(df['RainToday'].unique())

# print(df['RainToday'].value_counts())

# print(pd.get_dummies(df['RainToday'], drop_first=True).head())

# print(pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0))

# ---Explore problems within numerical variables---

""" numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :\n\n', numerical)

print(df[numerical].head())

print(df[numerical].isnull().sum())   

print(round(df[numerical].describe()), 2)
 """
""" plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2,2,2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2,2,4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

plt.show() """

""" IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
 """
# ---Declare feature vector and target variable---
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

if y.isnull().sum() > 0:
    y = y.fillna(y.mode()[0])


# ---Split data into separate training and test set---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print(X_train.shape, X_test.shape)

# ---Feature engineering---
# print(X_train.dtypes)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

# print(categorical)

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

# print(numerical)

# print(X_train[numerical].isnull().sum())

# print(X_test[numerical].isnull().sum())

""" for col in numerical:
    if X_train[col].isnull().mean() > 0:
        print(col, round(X_train[col].isnull().mean(), 4)) """

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col] = df1[col].fillna(col_median)

# print(X_train[numerical].isnull().sum())

# print(X_test[numerical].isnull().sum())

# print(X_train[categorical].isnull().mean())

""" for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean())) """

for df2 in [X_train, X_test]:
    df2['WindGustDir'] = df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0])
    df2['WindDir9am'] = df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0])
    df2['WindDir3pm'] = df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0])
    df2['RainToday'] = df2['RainToday'].fillna(X_train['RainToday'].mode()[0])

# print(X_train[categorical].isnull().sum())

# print(X_train.isnull().sum())

# print(X_test.isnull().sum())

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

# print(X_train.Rainfall.max(), X_test.Rainfall.max())

# print(X_train.Evaporation.max(), X_test.Evaporation.max())

# print(X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max())

# print(X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max())

# print(X_train[numerical].describe())

# ---Encode categorical variables---
# print(categorical)

# print(X_train[categorical].head())

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# print(X_train.head())

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
pd.get_dummies(X_train.Location), 
pd.get_dummies(X_train.WindGustDir),
pd.get_dummies(X_train.WindDir9am),
pd.get_dummies(X_train.WindDir3pm)], axis=1)

# print(X_train.head())

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
pd.get_dummies(X_test.Location), 
pd.get_dummies(X_test.WindGustDir),
pd.get_dummies(X_test.WindDir9am),
pd.get_dummies(X_test.WindDir3pm)], axis=1)

# print(X_test.head())

# ---Feature scaling---
# print(X_train.describe())

cols = X_train.columns

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

# print(X_train.describe())

# ---Model training---
logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(X_train, y_train)

# ---Predict results---
y_pred_test = logreg.predict(X_test)

x_test_pred_proba = logreg.predict_proba(X_test)

# print(y_pred_test)

# print(x_test_pred_proba[:,0])

# print(x_test_pred_proba[:,1])

# ---Check model accuracy---
""" print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

y_pred_train = logreg.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train))) """

# ---Check for overfitting and underfitting---
""" print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test))) """

""" logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

logreg100.fit(X_train, y_train)

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test))) """

""" logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)

logreg001.fit(X_train, y_train)

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test))) """

# ---Compare model accuracy with null accuracy---

""" print(y_test.value_counts())

null_accuracy = (22726/(22726+6366))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy)) """

# ---Confusion matrix---
cm = confusion_matrix(y_test, y_pred_test)

""" print('Confusion matrix\n\n', cm)

print('True Positives(TP) = ', cm[0,0])

print('True Negatives(TN) = ', cm[1,1])

print('False Positives(FP) = ', cm[0,1])

print('False Negatives(FN) = ', cm[1,0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

plt.show() """

# ---Classification metrices---
""" print(classification_report(y_test, y_pred_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

precision = TP / float(TP + FP)

print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

true_positive_rate = TP / float(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity)) """

# ---Adjusting the threshold---
y_pred_prob = logreg.predict_proba(X_test)[:10]

# print(y_pred_prob)

y_pred_prob_df = pd.DataFrame(y_pred_prob, columns=['Prob of - No Rain Tomorrow (0)', 'Prob of - Rain Tomorrow (1)'])

# print(y_pred_prob_df)

logreg.predict_proba(X_test)[:10, 1]

y_pred1 = logreg.predict_proba(X_test)[:, 1]

""" plt.rcParams['font.size'] = 12

plt.hist(y_pred1, bins=10)

plt.title('Histogram of predicted probabilities of rain')

plt.xlim(0,1)

plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')

plt.show() """

# ---Lower the threshold---

""" threshold = 0.5

cm1 = 0
    
y_pred1 = logreg.predict_proba(X_test)[:,1]
    
y_pred1 = y_pred1.reshape(-1,1)
    
y_pred2 = binarize(y_pred1.reshape(-1, 1), threshold).flatten()
    
y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
cm1 = confusion_matrix(y_test, y_pred2)
        
print ('With',threshold,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
        'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
        cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
        cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
        'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
        'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
        'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
        '====================================================', '\n\n')
     """

# ---ROC Curve---
""" fpr , tpr, thresholds = roc_curve(y_test, y_pred1, pos_label='Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show() """

# ---AUC---
""" ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC)) """

""" Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC)) """

# ---k-Fold Cross Validation---
""" scores = cross_val_score(logreg, X_train, y_train, cv=5, scoring='accuracy')

print('Cross-validation scores: {}'.format(scores))

print('Average cross-validation score: {:.4f}'.format(scores.mean())) """

# ---Hyperparameter optimization using GridSearch CV---
parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(estimator = logreg, param_grid = parameters, scoring = 'accuracy', cv = 5, verbose=3)

grid_search.fit(X_train, y_train)

print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

print('\n\nGridSearch CV best score : {:.4f}'.format(grid_search.best_score_))