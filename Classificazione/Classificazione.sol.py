# ESERCIZIO 1

# 1a
bcwds["diagnosis"].value_counts()

# 1b
bcwds["diagnosis"].value_counts().plot.pie();

# 1c
bcwds.iloc[:, 1:11].describe()

# 1d
bcwds["mean_area"].plot.hist(bins=20);

# 1e
bcwds.plot.scatter("mean_area", "mean_concave_pts");


# ESERCIZIO 2

is_malignant = X2d_val["mean_concave_pts"] > -0.0001 * X2d_val["mean_area"] + 0.15
y_pred = np.where(is_malignant, "M", "B")


# ESERCIZIO 3

# 3a
correct_class = y_pred == y_val

# 3b
correct_class.mean()


# ESERCIZIO 4

# 4a
model.score(X2dn_val, y_val)

# 4b
confusion_matrix(y_val, model.predict(X2dn_val))

# 4c
f1_score(y_val, model.predict(X2dn_val), pos_label="M")


# ESERCIZIO 5

# 5a
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="saga", penalty="l1", C=0.1))
])
model.fit(X, y)
pd.Series(model.named_steps["lr"].coef_[0], index=X.columns)

# 5b
model = Pipeline([
    ("scaler", None),
    ("lr", LogisticRegression(solver="saga"))
])
grid = [
    {
        "scaler": [None, StandardScaler()],
        "lr__penalty": ["none"]
    },
    {
        "scaler": [None, StandardScaler()],
        "lr__penalty": ["l2", "l1"],
        "lr__C": np.logspace(-2, 2, 5)
    },
    {
        "scaler": [None, StandardScaler()],
        "lr__penalty": ["elasticnet"],
        "lr__C": np.logspace(-2, 2, 5),
        "lr__l1_ratio": [0.2, 0.5]
    }
]
gs = GridSearchCV(model, grid, cv=skf)
gs.fit(X, y)
pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").head(5)