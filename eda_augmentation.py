# ── CELL 1: Install ───────────────────────────────────────────────────────────
!pip install nltk -q
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("omw-1.4")
nltk.download("punkt")


# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import random
import re
import pandas as pd
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

random.seed(42)


# ── CELL 3: EDA operations ────────────────────────────────────────────────────

def get_synonyms(word: str) -> list:
    """Get synonyms from WordNet, excluding the word itself."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(words: list, n: int) -> list:
    """Replace n random non-stopword words with a synonym."""
    STOPWORDS = {"the","a","an","is","are","was","were","be","been","being",
                 "have","has","had","do","does","did","will","would","shall",
                 "should","may","might","must","can","could","to","of","in",
                 "for","on","with","at","by","from","and","or","but","as"}
    new_words  = words.copy()
    candidates = [w for w in words if w.lower() not in STOPWORDS]
    random.shuffle(candidates)
    replaced = 0
    for word in candidates:
        syns = get_synonyms(word)
        if syns:
            new_words = [random.choice(syns) if w == word else w for w in new_words]
            replaced += 1
            if replaced >= n:
                break
    return new_words


def random_insertion(words: list, n: int) -> list:
    """Insert a synonym of a random word at a random position n times."""
    new_words = words.copy()
    for _ in range(n):
        syns = []
        attempts = 0
        while not syns and attempts < 10:
            syns = get_synonyms(random.choice(new_words))
            attempts += 1
        if syns:
            new_words.insert(random.randint(0, len(new_words)), random.choice(syns))
    return new_words


def random_swap(words: list, n: int) -> list:
    """Swap two random words n times."""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            i, j = random.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


def random_deletion(words: list, p: float = 0.1) -> list:
    """Randomly delete each word with probability p."""
    if len(words) == 1:
        return words
    new_words = [w for w in words if random.random() > p]
    return new_words if new_words else [random.choice(words)]


def eda(text: str, alpha: float = 0.1, num_aug: int = 1) -> list:
    """
    Apply all 4 EDA operations to generate num_aug augmented versions.

    alpha  : fraction of words to modify (0.1 = 10% of words)
    num_aug: number of augmented sentences to return
    """
    words  = text.split()
    n      = max(1, int(alpha * len(words)))    # num words to modify
    augmented = []

    ops = [synonym_replacement, random_insertion, random_swap]

    for i in range(num_aug):
        op = ops[i % len(ops)]                  # cycle through operations

        if op == random_deletion:
            new_words = random_deletion(words, p=alpha)
        else:
            new_words = op(words, n)

        augmented.append(" ".join(new_words))

    return augmented


# ── CELL 4: Sanity check ──────────────────────────────────────────────────────
sample = "The government allocated funds for rural infrastructure and agricultural development."
print("Original:", sample)
for i, aug in enumerate(eda(sample, num_aug=3), 1):
    print(f"  Aug {i}: {aug}")


# ── CELL 5: Load and split FIRST ──────────────────────────────────────────────
df = pd.read_csv("your_cleaned_file.csv")
df = df[["chunk_text", "summary_text"]].dropna().reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)

# Save val and test — never touched again
val_df.to_csv("val.csv",   index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("val.csv and test.csv locked.")


# ── CELL 6: Augment train ─────────────────────────────────────────────────────
# EDA on both chunk and summary keeps the pair aligned
# alpha=0.1 → modify 10% of words (gentle, meaning preserved)
# num_aug=2  → 127 rows becomes 381

NUM_AUGMENTS = 2
ALPHA        = 0.1

augmented_rows = []

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Augmenting"):
    chunk   = str(row["chunk_text"])
    summary = str(row["summary_text"])

    chunk_augs   = eda(chunk,   alpha=ALPHA, num_aug=NUM_AUGMENTS)
    summary_augs = eda(summary, alpha=ALPHA, num_aug=NUM_AUGMENTS)

    for c, s in zip(chunk_augs, summary_augs):
        augmented_rows.append({
            "chunk_text"  : c,
            "summary_text": s,
            "is_augmented": True,
        })

train_df["is_augmented"] = False
aug_df    = pd.DataFrame(augmented_rows)
train_aug = pd.concat([train_df, aug_df], ignore_index=True)
train_aug = train_aug.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nOriginal : {len(train_df)}")
print(f"Augmented: {len(aug_df)}")
print(f"Total    : {len(train_aug)}")

train_aug.to_csv("train_augmented.csv", index=False)
print("train_augmented.csv saved")


# ── CELL 7: Spot check ────────────────────────────────────────────────────────
sample_rows = train_aug[train_aug["is_augmented"]].head(3)
orig_rows   = train_aug[~train_aug["is_augmented"]].head(3)

for (_, aug_row), (_, orig_row) in zip(sample_rows.iterrows(), orig_rows.iterrows()):
    print("\nORIGINAL CHUNK  :", orig_row["chunk_text"][:150])
    print("AUGMENTED CHUNK :", aug_row["chunk_text"][:150])
    print("─" * 60)
