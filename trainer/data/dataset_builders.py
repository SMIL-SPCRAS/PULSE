import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_embedding_dict(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)
    keys = data["keys"]
    emb_dict = {}
    for i, k in enumerate(keys):
        emb = data[f"v_{i}"]
        emb_dict[str(k)] = emb
    return emb_dict

def get_embedding(emb_dict: Dict[str, np.ndarray], key: str) -> torch.Tensor:
    emb = emb_dict[key]
    emb = np.array(emb)
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]
    elif emb.ndim != 2:
        raise ValueError(f"Ожидал (1,T,H) или (T,H), а получил {emb.shape} для key={key}")
    return torch.from_numpy(emb).float()

class CMUMOSEIEmotionDataset(Dataset):

    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        emotion_columns: Optional[List[str]] = None,
        dataset_name: str = "cmu_mosei",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)

        self.key_column = "video_name"
        if emotion_columns is None:
            emotion_columns = [
                "Neutral", "Anger", "Disgust", "Fear",
                "Happiness", "Sadness", "Surprise"
            ]
        self.emotion_columns = emotion_columns

        valid_rows = []
        for _, row in self.df.iterrows():
            key = str(row[self.key_column])
            if key in self.emb_dict:
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "CMUMOSEIEmotionDataset: no matches."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = str(row[self.key_column])

        x = get_embedding(self.emb_dict, key)
        length = x.shape[0]

        vals = pd.to_numeric(row[self.emotion_columns], errors="coerce").astype("float32").to_numpy()
        emo_vec = torch.from_numpy(vals.copy())

        return {
            "x": x,
            "length": length,
            "emotion_mosei": emo_vec,
            "emotion_resd": None,
            "personality": None,
            "ah": None,
            "dataset_name": self.dataset_name,
        }

class RESDEmotionDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        dataset_name: str = "resd",
        label_column: str = "emotion",
        label_mapping: Dict[str, int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)
        self.key_column = "name"
        self.label_column = label_column

        if label_mapping is None:
            raise ValueError("RESDEmotionDataset: label_mapping is none.")
        self.label_mapping = label_mapping

        valid_rows = []
        for _, row in self.df.iterrows():
            base_name = str(row[self.key_column])
            key = base_name
            if key in self.emb_dict:
                row = row.copy()
                row["_emb_key"] = key
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "RESDEmotionDataset: no matches."
            )

        unique_labels = sorted(self.df[self.label_column].unique())
        missing = [str(lbl) for lbl in unique_labels if str(lbl) not in self.label_mapping]
        if missing:
            raise ValueError(f"В label_mapping no class from csv: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = row["_emb_key"]

        x = get_embedding(self.emb_dict, key)
        length = x.shape[0]

        raw_label = str(row[self.label_column])
        label = self.label_mapping[raw_label]

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": label,
            "personality": None,
            "ah": None,
            "dataset_name": self.dataset_name,
        }

class FIV2PersonalityDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        subset: Optional[str],
        dataset_name: str = "fiv2",
        personality_columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)

        if subset is not None:
            subset = subset.lower()
            aliases = {
                "train": {"train"},
                "test": {"test"},
                "validation": {"validation", "val", "valid", "dev"},
                "val": {"validation", "val", "valid", "dev"},
                "valid": {"validation", "val", "valid", "dev"},
                "dev": {"validation", "val", "valid", "dev"},
            }
            if subset not in aliases:
                raise ValueError("subset must be 'train', 'test', 'validation', 'val', 'valid', 'dev', or None")
            if "Subset" not in self.df.columns:
                raise ValueError("FIV2 labels CSV has no 'Subset' column; use subset=None for a split-specific CSV")
            subset_values = self.df["Subset"].astype(str).str.lower()
            self.df = self.df[subset_values.isin(aliases[subset])].reset_index(drop=True)

        self.key_column = "NAME_VIDEO"
        if personality_columns is None:
            personality_columns = [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "non-neuroticism",
            ]
        self.personality_columns = personality_columns

        valid_rows = []
        for _, row in self.df.iterrows():
            raw_name = str(row[self.key_column])
            file_id = os.path.splitext(raw_name)[0]
            if file_id in self.emb_dict:
                row = row.copy()
                row["_emb_key"] = file_id
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                f"FIV2PersonalityDataset({subset}): no matches."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = row["_emb_key"]

        x = get_embedding(self.emb_dict, key)
        length = x.shape[0]

        vals = pd.to_numeric(
            row[self.personality_columns],
            errors="coerce"
        ).astype("float32").to_numpy()
        pers = torch.from_numpy(vals.copy())

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": None,
            "personality": pers,
            "ah": None,
            "dataset_name": self.dataset_name,
        }

def _parse_bah_txt_lines(
    txt_path: str,
    sep: str = ",",
    id_field: int = 0,
    label_field: int = 1,
) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < max(id_field, label_field) + 1:
                raise ValueError(f"Bad line in {txt_path}: '{line}'")
            sid = parts[id_field].strip()
            y = int(parts[label_field].strip())
            pairs.append((sid, y))
    return pairs

def _match_bah_key(emb_dict: Dict[str, np.ndarray], key: Any) -> str:
    k = str(key).strip()
    variants = []

    def add(x: str):
        if x and x not in variants:
            variants.append(x)

    add(k)
    add(k.replace("\\", "/"))

    norm = k.replace("\\", "/")
    if norm.startswith("Videos/"):
        add("Audio/" + norm[len("Videos/"):])
    elif norm.startswith("Audio/"):
        add("Videos/" + norm[len("Audio/"):])

    base = os.path.basename(norm)
    stem = os.path.splitext(base)[0]

    add(base)
    add(stem)

    for v in variants:
        if v in emb_dict:
            return v

    raise KeyError(f"Embedding id '{key}' not found in NPZ")

class BAHAbsencePresenceDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_txt_path: str,
        dataset_name: str = "bah",
        sep: str = ",",
        id_field: int = 0,
        label_field: int = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)
        self.samples = _parse_bah_txt_lines(
            labels_txt_path,
            sep=sep,
            id_field=id_field,
            label_field=label_field,
        )

        if len(self.samples) == 0:
            raise RuntimeError("BAHAbsencePresenceDataset: empty txt labels.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_key, label = self.samples[idx]
        key = _match_bah_key(self.emb_dict, raw_key)

        x = get_embedding(self.emb_dict, key)
        length = x.shape[0]

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": None,
            "personality": None,
            "ah": int(label),
            "dataset_name": self.dataset_name,
        }
