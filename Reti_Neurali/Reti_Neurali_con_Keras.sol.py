# ESERCIZIO 1

# 1a (1*32 + 32 + 32*1 + 1 = 97 parametri)
model = Sequential([
    Dense(32, activation="relu", input_dim=1),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=3, epochs=10)
r2_score(y_val, model.predict(X_val))

# 1b (1*32 + 32 + 32*8 + 8 + 8*1 + 1 = 337 parametri)
model = Sequential([
    Dense(32, activation="relu", input_dim=1),
    Dense(8, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=3, epochs=10)
r2_score(y_val, model.predict(X_val))


# ESERCIZIO 2

# 2a (2*32 + 32 + 32*2 + 2 = 162 parametri)
model = Sequential([
    Dense(32, activation="relu", input_dim=2),
    Dense(2, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
fit_history = model.fit(X_train, yt_train, batch_size=10, epochs=10)
model.evaluate(X_val, yt_val)

# 2b (2*32 + 32 + 32*8 + 8 + 8*2 + 2 = 378 parametri)
model = Sequential([
    Dense(32, activation="relu", input_dim=2),
    Dense(8, activation="relu"),
    Dense(2, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
fit_history = model.fit(X_train, yt_train, batch_size=10, epochs=10)
model.evaluate(X_val, yt_val)