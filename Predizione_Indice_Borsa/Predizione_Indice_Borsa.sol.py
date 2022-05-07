# ESERCIZIO 1

# 1a
data.describe()

# 1b
hlm = (high + low) / 2

# 1c
hlm.describe()

# 1d
hlm.groupby(hlm.index.year).describe()


# ESERCIZIO 2

# 2a
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_val, y_val)

# 2b
model = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("regr", Ridge(alpha=1))
])
model.fit(X_train, y_train)
model.score(X_val, y_val)

# 2c
model = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scale", StandardScaler()),
    ("regr", LinearRegression())
])
model.fit(X_train, y_train)
model.score(X_val, y_val)


# ESERCIZIO 3

def gain(C, C_pred):
    O = open.reindex_like(C)
    CO_diff = C - O
    growth = C_pred > O
    decline = C_pred < O
    return CO_diff[growth].sum() - CO_diff[decline].sum()


# ESERCIZIO 4

def prepare_data(features, target):
    X = pd.DataFrame(features)
    X.dropna(inplace=True)
    y = target.reindex_like(X)
    return X, y


# ESERCIZIO 5

def gain(D, D_pred):
    growth = D_pred > 0
    decline = D_pred < 0
    return D[growth].sum() - D[decline].sum()

def roi(D, D_pred):
    mean_open = open.reindex_like(D).mean()
    return gain(D, D_pred) / mean_open


# ESERCIZIO 6

rois = []
for s in range(1000):
    np.random.seed(s)
    preds = np.random.normal(y_train.mean(), y_train.std(), len(y_val))
    rois.append(roi(y_val, preds))
print(np.mean(rois))


# ESERCIZIO 7

model = Pipeline([
    ("scale", None),
    ("regr", KernelRidge(kernel="rbf"))
])
grid = {
    "scale": [None, StandardScaler()],
    "regr__gamma": [0.001, 0.01, 0.1],
    "regr__alpha": np.logspace(-3, 2, 6)
}
gs = GridSearchCV(model, grid, scoring=roi_scorer, cv=tss)
gs.fit(X, y)
cv_results = pd.DataFrame(gs.cv_results_)

cv_results.sort_values("mean_test_score", ascending=False).head(5)


# ESERCIZIO 8

# 8a
from sklearn.linear_model import Lasso
model = Lasso(alpha=3)
model.fit(X, y)
pd.Series(model.coef_, X.columns)

# 8b
cross_validate(LinearRegression(), X, y, cv=tss, scoring=roi_scorer)

# 8c
model = Pipeline([
    ("scale", None),
    ("regr", KernelRidge(kernel="rbf"))
])
grid = {
    "scale": [None, StandardScaler()],
    "regr__gamma": [0.001, 0.01, 0.1],
    "regr__alpha": np.logspace(-3, 2, 6)
}
gs = GridSearchCV(model, grid, cv=tss, scoring=roi_scorer)
gs.fit(X, y)
cv_results = pd.DataFrame(gs.cv_results_)

cv_results.sort_values("mean_test_score", ascending=False).head(5)