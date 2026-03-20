import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from pathlib import Path

def split_dataset(annotation_csv, output_dir, seed=42):
    df = pd.read_csv(annotation_csv)

    # Train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["category"],
        random_state=seed
    )

    # Validation vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["category"],
        random_state=seed
    )

    output_dir = Path(output_dir)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    return train_df, val_df, test_df