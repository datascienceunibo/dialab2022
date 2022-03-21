# ESERCIZIO 1

# 1a
data_summer.describe()

# 1b
data_summer["temp"].plot.hist(bins=20);

data_summer["demand"].plot.hist(bins=20);

# 1c
data_summer.plot.scatter("temp", "demand");


# ESERCIZIO 2

# 2a
ex_model = make_model(0.15, -1)

# 2b
ex_model(np.array([20, 25, 30]))

# 2c
plot_model_on_data(temp, demand, ex_model)

# 2d
np.mean(np.square(ex_model(temp) - demand))
# MSE più elevato -> minore accuratezza


# ESERCIZIO 3

# 3a
for iteration in range(20):
    # esegui un passo di discesa e aggiorna i parametri
    alpha, beta = ulr_gd_step(temp, demand, alpha, beta, 0.001)
    # salva i valori correnti dei parametri
    alpha_vals.append(alpha)
    beta_vals.append(beta)

# 3b
alpha, beta

# 3c
gd_model = make_model(alpha, beta)

# 3d
plot_model_on_data(temp, demand, gd_model)

# 3e
np.mean(np.square(gd_model(temp) - demand))
# oppure
ulr_mse(temp, demand, alpha, beta)


# ESERCIZIO 4

# 4a
data_winter = data.loc[(data.index.month <= 2) | (data.index.month >= 12)]

# 4b
data_winter.plot.scatter("temp", "demand");

# 4c
# estrarre i vettori di valori x e y
x, y = data_winter["temp"].values, data_winter["demand"].values
# inizializzare i parametri alpha e beta
alpha, beta = 0, 0
# eseguire 300 iterazioni di discesa gradiente
for it in range(300):
    alpha, beta = ulr_gd_step(x, y, alpha, beta, 0.01)
# creare il modello con i parametri finali
winter_model = make_model(alpha, beta)

# 4d
plot_model_on_data(x, y, winter_model)

# 4e
winter_model(-5)


# ESERCIZIO 5

# 5a
theta = np.zeros(X1.shape[1])

# 5b
theta_vals = [theta]

# 5c
for iteration in range(50):
    theta = lr_gd_step(X1, y, theta, 0.000001)
    theta_vals.append(theta)

# 5d
np.mean(np.square(X1 @ theta - y))