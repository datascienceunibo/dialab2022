# ESERCIZIO 1

# 1a
tips.loc[0, "size"]

# 1b
tips.loc[tips["total_bill"].idxmax()]

# 1c
tips["tip"].mean()

# 1d
(tips["total_bill"] / tips["size"]).max()


# ESERCIZIO 2

# 2a
(tips["total_bill"] / tips["size"]).describe()

# 2b
pd.cut(tips["tip"] / tips["total_bill"], 3).value_counts()


# ESERCIZIO 3

# 3a
pd.cut(tips["total_bill"], 3).value_counts().plot.pie();

# 3b
tips.loc[tips["time"] == "Dinner", "day"].value_counts().plot.pie();

# 3c
(tips["tip"] / tips["total_bill"]).plot.hist();

# 3d
(tips["tip"] / tips["total_bill"]).plot.box();

# 3e
tips.plot.scatter("tip", "size");

# 3f
for n, day in enumerate(["Thur", "Fri", "Sat", "Sun"], start=1):
    tips.loc[tips["day"] == day, "smoker"].value_counts().plot.bar(ax=plt.subplot(2, 2, n), title=day)


# ESERCIZIO 4

# 4a
tips.groupby("smoker")["size"].mean()

# 4b
tips.groupby("day")["tip"].agg(["sum", "mean"])

# 4c
tips.loc[tips["day"] == "Fri"].groupby("time")["tip"].mean()

# 4d
tips.groupby(tips["total_bill"] >= 20)["tip"].mean()


# ESERCIZIO 5

# 5a
tips.groupby(["day", "time"]).size().unstack(fill_value=0)

# 5b
tips.pivot_table(
    values="tip",
    index="day",
    columns="time",
    aggfunc="sum",
    fill_value=0,
)

# 5c
tips.pivot_table(
    values=["total_bill", "tip"],
    index=pd.cut(tips["total_bill"], 3),
    columns="time",
)