# ESERCIZIO 1

# 1a
len(amazon_train)

# 1b
amazon_train["user"].nunique()

# 1c
amazon_train["item"].nunique()

# 1d
amazon_train["rating"].mean()

# 1e
amazon_train["user"].value_counts().min()

# 1f
amazon_train["item"].value_counts().idxmax()

# 1g
mean_ratings = amazon_train.groupby("item")["rating"].mean()
mean_ratings.sort_values(ascending=False).head(10)
# o in breve: mean_ratings.nlargest(10)


# ESERCIZIO 2

# 2a
P.sum()

# 2b
P.shape[0]

# 2c
P.shape[1]

# 2d
R.mean(where=P)
# oppure: R[P].mean()

# 2e
P.sum(1).min()

# 2f
train_ratings.columns[P.sum(0).argmax()]

# 2g
train_ratings.columns[(-R.mean(0, where=P)).argsort()[:10]]
# oppure, se il parametro where non è disponibile:
# train_ratings.columns[(-R.sum(0) / P.sum(0)).argsort()[:10]]


# ESERCIZIO 3

def predict_from_all(u, i):
    voters = list(P[:, i].nonzero()[0])
    predicted_vote = (cosim[u, voters] @ R[voters, i]) / cosim[u, voters].sum()
    return predicted_vote if not np.isnan(predicted_vote) else R.mean(where=P)


# ESERCIZIO 4

# 4a
def RMSE(actual, predicted):
    return np.sqrt(np.mean(np.square(predicted - actual)))

# 4b
val_predictions = get_val_predictions(predict_from_all)

# 4c
RMSE(val_actual, val_predictions)


# ESERCIZIO 5

# 5a
def predict_from_neighbors(u, i):
    voters = list(P[:, i].nonzero()[0])
    voters.sort(key=lambda x: cosim[u, x], reverse=True)
    voters = voters[:k]
    predicted_vote = (cosim[u, voters] @ R[voters, i]) / cosim[u, voters].sum()
    return predicted_vote if not np.isnan(predicted_vote) else R[P].mean()

# 5b
RMSE(val_actual, get_val_predictions(predict_from_neighbors))


# ESERCIZIO 6

# 6a
ubr = KNNBasic(k=10, sim_options={"name": "cosine"})
ubr.fit(trainset)

# 6b
preds = ubr.test(valset)

rmse(preds)

mae(preds)


# ESERCIZIO 7

# 7a
grid = {"n_factors": range(3, 31, 3)}
gs = GridSearchCV(SVD, grid, cv=kf, refit=True)
gs.fit(train_dataset)

gs.best_params["rmse"]["n_factors"]

# 7b
preds = gs.test(valset)
rmse(preds), mae(preds)


# ESERCIZIO 8

# 8a
preds = [
    gs.predict(target_user, trainset.to_raw_iid(ii))
    for ii in range(trainset.n_items)
]

# 8b
for p in sorted(preds, key=lambda p: p.est, reverse=True)[:10]:
    print(p.iid)

# 8c
def recommend(user_id, n_recomms):
    preds = [
        gs.predict(user_id, trainset.to_raw_iid(ii))
        for ii in range(trainset.n_items)
    ]
    preds.sort(key=lambda p: p.est, reverse=True)
    return [p.iid for p in preds[:n_recomms]]

# test
recommend(target_user, 10)