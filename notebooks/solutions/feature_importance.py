data = pd.read_csv("data/adult.csv", index_col=0)
y = data.income.values
X = pd.get_dummies(data.drop("income", axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler().fit(X_train)
X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)

LogisticRegression().fit(X_train_, y_train).score(X_test_, y_test)

print(X_train.shape)

select = SelectFromModel(RandomForestClassifier(), threshold="5 * median")
X_train_selected = select.fit_transform(X_train_, y_train)
X_test_selected = select.transform(X_test_)

score = LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected, y_test)
print(score)

print(X_train_selected.shape)

poly = PolynomialFeatures(degree=2).fit(X_train_selected)
X_train_selected_poly = poly.transform(X_train_selected)
X_test_selected_poly = poly.transform(X_test_selected)

lr = LogisticRegression(C=0.01, penalty="l1").fit(X_train_selected_poly, y_train)
print(lr.score(X_test_selected_poly, y_test))

np.array(poly.get_feature_names(X.columns[select.get_support()]))[lr.coef_.ravel() != 0]
