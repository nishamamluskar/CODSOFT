# Movie Dataset – README

## Overview

This dataset contains movie information in a custom text format, where each line stores four fields:

1. **Movie ID**
2. **Movie Title**
3. **Genre**
4. **Movie Description**

Each field is separated using **triple colons (`:::`)**.
Example:

```
1 ::: Oscar et la dame rose (2009) ::: drama ::: Listening in to a conversation...
2 ::: Cupid (1997) ::: thriller ::: A brother and sister...
3 ::: Young, Wild and Wonderful (1980) ::: adult ::: As the bus empties...
```

This format is suitable for content-based recommendation systems where genres or descriptions are used to compute similarity between movies.

---

## File Format

* **Filename:** movies.txt
* **Encoding:** UTF-8
* **Delimiter:** `:::` (triple colon with variable spacing)
* **Lines:** One movie record per line
* **Number of Columns:** 4
* **Structure:**

  ```
  <id> ::: <title> ::: <genre> ::: <description>
  ```

---

## How to Load the Dataset in Python

Use the following code to load the dataset safely:

```python
import pandas as pd

df = pd.read_csv("movies.txt", sep=r"\s*:::\s*", engine="python", header=None)
df.columns = ["id", "title", "genre", "description"]
```

The regular expression `\s*:::\s*` ensures that the file loads correctly even if spacing around the delimiter varies.

---

## Building a Genre-Based Movie Recommendation System

### 1. Vectorize the Genre Text

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["genre"])
```

### 2. Compute Similarity Between Movies

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

### 3. Create the Recommendation Function

```python
def recommend(title, n=5):
    if title not in df["title"].values:
        return ["Movie not found"]
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return df["title"].iloc[[i[0] for i in scores]].tolist()
```

### 4. Example Usage

```python
print(recommend("Oscar et la dame rose (2009)"))
```

---

## Applications

* Content-based movie recommendation systems
* Genre clustering and visualization
* Text mining on movie descriptions
* NLP feature extraction (TF-IDF, embeddings)

---

## Notes

* Descriptions are long and useful for more advanced recommenders (semantic similarity, embedding models).
* Genre column works best for simple TF-IDF-based recommendations.
* Ensure the file remains UTF-8 encoded to avoid parsing errors.

---

## License

Dataset usage depends on original source license (not included in this file).

---

If you need a **README with screenshots**, **a PDF version**, or one tailored for GitHub**, tell me—I can generate it.
