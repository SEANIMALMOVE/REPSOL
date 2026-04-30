from pathlib import Path

import pandas as pd  # type: ignore
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore


def _validate_label_column(df: pd.DataFrame, label_col: str) -> None:
    if label_col not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Column '{label_col}' not found. Available columns: {available}")


def create_test_and_cv_splits(
    annotation_csv,
    output_dir,
    seed: int = 42,
    label_col: str = "category",
    test_ratio: float = 0.15,
    min_test_per_class: int = 10,
    n_splits: int = 5,
):
    """
    Create a fixed stratified test set and 5-fold stratified CV splits for tuning.

    Outputs:
    - output_dir/test.csv
    - output_dir/trainval.csv
    - output_dir/cv_folds/fold_{k}_train.csv
    - output_dir/cv_folds/fold_{k}_val.csv
    - output_dir/cv_folds/fold_summary.csv
    """
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be in (0, 1).")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    df = pd.read_csv(annotation_csv)
    _validate_label_column(df, label_col)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "cv_folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    class_counts = df[label_col].value_counts().sort_index()
    test_parts = []
    unmet_minimum = []

    # Per-class allocation gives better control when classes are imbalanced.
    for i, class_name in enumerate(class_counts.index):
        class_df = df[df[label_col] == class_name]
        n_class = len(class_df)

        requested = max(int(round(n_class * test_ratio)), min_test_per_class)
        # Keep enough samples so CV can still build n_splits folds for this class.
        max_allowed = max(1, n_class - n_splits)
        test_n = min(requested, max_allowed)

        if test_n < min_test_per_class:
            unmet_minimum.append(
                {
                    "class": class_name,
                    "class_total": n_class,
                    "requested_min_test": min_test_per_class,
                    "actual_test": test_n,
                }
            )

        test_sample = class_df.sample(n=test_n, random_state=seed + i)
        test_parts.append(test_sample)

    test_df = pd.concat(test_parts, ignore_index=False)
    trainval_df = df.drop(index=test_df.index)

    # Guardrail: StratifiedKFold requires each class in trainval to have >= n_splits samples.
    trainval_counts = trainval_df[label_col].value_counts()
    too_small = trainval_counts[trainval_counts < n_splits]
    if not too_small.empty:
        raise ValueError(
            "Some classes are too small for StratifiedKFold after test split. "
            f"Need >= {n_splits} samples per class in trainval. "
            f"Too small: {too_small.to_dict()}"
        )

    test_df = test_df.reset_index(drop=True)
    trainval_df = trainval_df.reset_index(drop=True)

    test_df.to_csv(output_dir / "test.csv", index=False)
    trainval_df.to_csv(output_dir / "trainval.csv", index=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(trainval_df, trainval_df[label_col]),
        start=1,
    ):
        fold_train = trainval_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = trainval_df.iloc[val_idx].reset_index(drop=True)

        fold_train.to_csv(folds_dir / f"fold_{fold_idx}_train.csv", index=False)
        fold_val.to_csv(folds_dir / f"fold_{fold_idx}_val.csv", index=False)

        val_class_min = int(fold_val[label_col].value_counts().min())
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_size": len(fold_train),
                "val_size": len(fold_val),
                "min_class_count_in_val": val_class_min,
            }
        )

    fold_summary_df = pd.DataFrame(fold_rows)
    fold_summary_df.to_csv(folds_dir / "fold_summary.csv", index=False)

    print("Saved fixed test + CV folds to:", output_dir)
    print("Train+Val pool:", len(trainval_df))
    print("Test:", len(test_df))
    print(f"CV folds: {n_splits}")

    if unmet_minimum:
        print("\nClasses where min_test_per_class could not be fully met:")
        print(pd.DataFrame(unmet_minimum).to_string(index=False))

    return trainval_df, test_df, fold_summary_df, pd.DataFrame(unmet_minimum)


def split_dataset(annotation_csv, output_dir, seed=42):
    """
    Backward-compatible single split (70/15/15) used by existing code.
    """
    df = pd.read_csv(annotation_csv)
    _validate_label_column(df, "category")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["category"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["category"],
        random_state=seed,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    return train_df, val_df, test_df