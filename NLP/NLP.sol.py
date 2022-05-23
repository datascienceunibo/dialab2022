# ESERCIZIO 1

# 1a
reviews["stars"].value_counts()

# 1b
reviews["stars"].value_counts().plot.pie();

# 1c
reviews["text"].str.len().plot.hist(bins=20);


# ESERCIZIO 2

# 2a
reviews["label"] = np.where(reviews["stars"] >= 4, "pos", "neg")

# 2b
reviews["label"].value_counts()


# ESERCIZIO 3

# 3a
dtm_new = vect.transform(new_reviews)

# 3b
lrm.predict(dtm_new)

# 3c
lrm.predict_proba(dtm_new)


# ESERCIZIO 4

# 4a
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression(solver="saga", C=10))
])

# 4b
model.fit(reviews_train["text"], reviews_train["label"]);

# 4c
model.score(reviews_val["text"], reviews_val["label"])

# 4d
pd.Series(
    model.named_steps["classifier"].coef_[0],
    index=model.named_steps["vectorizer"].get_feature_names()
).nlargest(5)


# ESERCIZIO 5

# 5a
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(solver="saga", C=10))
])
grid = {
    "vectorizer__min_df": [3, 5, 10],
    "vectorizer__stop_words": [None, stoplist]
}
skf = StratifiedKFold(3, shuffle=True)
gs = GridSearchCV(model, grid, cv=skf)
gs.fit(reviews_train["text"], reviews_train["label"]);

# 5b
gs.score(reviews_val["text"], reviews_val["label"])


# ESERCIZIO 6

# 6a
def tokenize_with_pos(text):
    return nltk.pos_tag(nltk.tokenize.word_tokenize(text))

# 6b
model = Pipeline([
    ("vectorizer", TfidfVectorizer(min_df=3, tokenizer=tokenize_with_pos)),
    ("classifier", LogisticRegression(solver="saga", C=10))
])
model.fit(reviews_train["text"], reviews_train["label"]);

# 6c
len(model.named_steps["vectorizer"].get_feature_names())

# 6d
model.score(reviews_val["text"], reviews_val["label"])


# ESERCIZIO 7

# 7a
def tokenize_with_stemming(text):
    return [ps.stem(token) for token in nltk.word_tokenize(text)]
    # oppure: return list(map(ps.stem, nltk.word_tokenize(text)))

# 7b
model = Pipeline([
    ("vectorizer", TfidfVectorizer(min_df=3, tokenizer=tokenize_with_stemming)),
    ("classifier", LogisticRegression(solver="saga", C=10))
])
model.fit(reviews_train["text"], reviews_train["label"]);

# 7c
len(model.named_steps["vectorizer"].get_feature_names())

# 7d
model.score(reviews_val["text"], reviews_val["label"])


# ESERCIZIO 8

# 8a
def label_review(review):
    sentences = nltk.sent_tokenize(review)
    scores = list(map(vader.polarity_scores, sentences))
    return "pos" if sum(s["compound"] for s in scores) >= 0 else "neg"

# 8b
vader_preds = [label_review(review) for review in reviews_val["text"]]

# 8c
accuracy_score(reviews_val["label"], vader_preds)