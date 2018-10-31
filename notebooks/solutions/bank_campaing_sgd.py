import pandas as pd
data = pd.read_csv("data/bank-campaign.csv")

data.head()

y = data.target
X = data.drop('target', axis=1).astype(float)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.pipeline import make_pipeline
lr = make_pipeline(StandardScaler(), LogisticRegression())

cross_val_score(lr, X_train, y_train, cv=5)

sgd = make_pipeline(StandardScaler(), SGDClassifier(loss='log', alpha=.01, max_iter=10, tol=1e-2))

cross_val_score(sgd, X_train, y_train, cv=5)

cross_val_score(SGDClassifier(alpha=.01, max_iter=10000, tol=1e-3), X_train, y_train, cv=5)