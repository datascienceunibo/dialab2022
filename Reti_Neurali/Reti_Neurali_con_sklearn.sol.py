# ESERCIZIO 1

# 1a
model = LinearRegression()
model.fit(X_train, y_train);

# 1b
model.score(X_val, y_val)

# 1c
plot_model_on_data(X_val, y_val, model)


# ESERCIZIO 2

# 2a
model = LogisticRegression(solver="saga")
model.fit(X_train, y_train);

# 2b
model.score(X_val, y_val)

# 2c
y_pred = model.predict(X_val)
plt.scatter(*X_val.T, s=3, c=np.where(y_pred, "red", "blue"));


# ESERCIZIO 3

# dalle variabili di input calcoliamo quelle del primo strato nascosto
H1_val = relu(X_val @ model.coefs_[0] + model.intercepts_[0])

# da queste calcoliamo quelle del secondo strato nascosto
H2_val = relu(H1_val @ model.coefs_[1] + model.intercepts_[1])

# generiamo il grafico, adattando il codice usato sopra
plt.scatter(*H2_val.T, s=5, c=np.where(y_val, "red", "blue"))
xlim, ylim = plt.xlim(), plt.ylim()
w = model.coefs_[2].T[0]
b = model.intercepts_[2]
sep_x = np.linspace(*xlim, 2)
sep_y = -w[0]/w[1]*sep_x -b/w[1]
plt.plot(sep_x, sep_y, c="green", lw=3);
plt.xlim(xlim); plt.ylim(ylim);


# ESERCIZIO 4

# 4a
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(activation="relu"))
])
grid = {
    "mlp__hidden_layer_sizes": [16, 32, (16, 8)],
    "mlp__batch_size": [100, 200]
}
skf = StratifiedKFold(3, shuffle=True)
gs = GridSearchCV(model, grid, cv=skf)
gs.fit(X_train, y_train);

# 4b
gs.best_params_

# 4c
gs.score(X_test, y_test)