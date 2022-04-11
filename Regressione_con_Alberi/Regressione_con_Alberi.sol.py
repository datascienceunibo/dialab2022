# ESERCIZIO 1

# 1a
X_train_num = data_train[numeric_vars + binary_vars]
X_val_num = data_val[numeric_vars + binary_vars]

# 1b
model = Ridge()
model.fit(X_train_num, y_train)
model.score(X_val_num, y_val)

# 1c
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", Ridge())
])
model.fit(X_train_num, y_train)
model.score(X_val_num, y_val)


# ESERCIZIO 2

model = Pipeline([
    ("encoder", OneHotEncoder()),
    ("regr",    Ridge())
])
model.fit(X_train_cat, y_train)
model.score(X_val_cat, y_val)


# ESERCIZIO 3

# 3a
model = Pipeline([
    ("preproc", ColumnTransformer([
        ("numeric", PolynomialFeatures(include_bias=False), numeric_vars + binary_vars),
        ("categorical", OneHotEncoder(), categorical_vars)
    ])),
    ("regr" , Ridge())
])
grid = {
    "preproc__numeric__degree": [1, 2, 3],
    "regr__alpha": [0.01, 1]
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(data_train_sample, y_train_sample)
gs.best_params_

gs.score(data_val, y_val)

# 3b
model = Pipeline([
    ("preproc", ColumnTransformer([
        ("numeric", Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False))
        ]), numeric_vars + binary_vars),
        ("categorical", OneHotEncoder(), categorical_vars)
    ])),
    ("regr" , Ridge())
])
grid = {
    "preproc__numeric__poly__degree": [1, 2, 3],
    "regr__alpha": [0.01, 1]
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(data_train_sample, y_train_sample)
gs.best_params_

gs.score(data_val, y_val)