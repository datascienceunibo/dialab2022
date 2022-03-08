# ESERCIZIO 1

# 1a
population[4]

# 1b
states[-3:]

# 1c
population[states == "Florida"]

# 1d
# promemoria: gli underscore "_" possono essere inseriti
# liberamente nei valori numerici per migliore leggibilità
states[population >= 20_000_000]

# 1e
population.sum()

# 1f
states[population.argmin()]


# ESERCIZIO 2

# 2a
area_km2 = area * 2.59
# stampo i primi 3 elementi
area_km2.head(3)

# 2b
density = population / area_km2
# stampo i primi 3 elementi
density.head(3)


# ESERCIZIO 3

# 3a
density[area.idxmin()]

# 3b
(population > 1_000_000).sum()

# 3c
population[west_coast].sum()

# 3d
density[population >= 10_000_000].mean()


# ESERCIZIO 4

# 4a
census["area"].max()

# 4b
state_to_state["Arizona"].sum()

# 4c
state_to_state.sum(axis=1).idxmin()


# ESERCIZIO 5

# 5a
census.loc["California", "area"]

# 5b
census.iloc[12, 0]

# 5c
census.loc[census["area"].idxmin(), "density"]

# 5d
census.loc["M":"N", "population"].sum()

# 5e
census.loc[census["population"] >= 20_000_000, "area"].sum()

# 5f
census.loc[census["from_abroad"] / census["population"] >= 0.01, "population"].mean()

# 5g
census.sort_values("density").head(5)["area"].mean()

# 5h
census.sort_values("area", ascending=False).iloc[2, 0]