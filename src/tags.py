# src/tags.py
import pandas as pd
from collections import Counter
from .config import IG_TAGS, CG_TAGS, CH_TAGS
from sklearn.preprocessing import MultiLabelBinarizer

# Parse tags into list
def parse_tags(df: pd.DataFrame) -> pd.DataFrame:
    df["tag_list"] = df["tags"].fillna("").apply(
        lambda s: [t.strip() for t in s.split(",") if t.strip()]
    )
    return df


#  Add binary columns for using MultiLabelBinarizer from sklearn
def add_tag_binaries(df: pd.DataFrame) -> pd.DataFrame:
    mlb = MultiLabelBinarizer(classes=IG_TAGS + CG_TAGS + CH_TAGS)
    binary_matrix = mlb.fit_transform(df["tag_list"])
    
    # Build DataFrame with proper column names
    binary_df = pd.DataFrame(
        binary_matrix,
        columns=[f"tag_{tag.replace(' ', '_')}" for tag in mlb.classes_],
        index=df.index
    )
    return pd.concat([df, binary_df], axis=1)


# Derive IG_state, CG_state & CH_state from tags
def derive_states(df: pd.DataFrame) -> pd.DataFrame:
    def ig_state(L):
        for tag in IG_TAGS:
            if tag in L:
                return tag
        return "IG unknown"

    def cg_state(L):
        for tag in CG_TAGS:
            if tag in L:
                return tag
        return "CG unknown"
    
    def ch_state(L):
        for tag in CH_TAGS:
            if tag in L:
                return tag
        return "CH normal"

    df["IG_state"] = df["tag_list"].apply(ig_state)
    df["CG_state"] = df["tag_list"].apply(cg_state)
    df["CH_state"] = df["tag_list"].apply(ch_state)
    return df


# Tag frequencies
def tag_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    all_tags = [t for L in df["tag_list"] for t in L]
    counts = Counter(all_tags)
    return (
        pd.DataFrame(counts.items(), columns=["Tag", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


# Combination counts (exact matches, order matters)
def combo_counts(df: pd.DataFrame) -> pd.DataFrame:
    cc = df["tags"].value_counts().reset_index()
    cc.columns = ["Tag combination", "Count"]
    return cc

def tag_events(df: pd.DataFrame) -> pd.DataFrame:
    # Convenience wrapper: parse -> binaries -> states
    df = parse_tags(df)
    df = add_tag_binaries(df)
    df = derive_states(df)
    return df