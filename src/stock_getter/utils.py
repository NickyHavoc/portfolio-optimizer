import pandas as pd
from pathlib import Path


def load_df_from_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_df_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=True)
