from data.dataset_builders import (
    CMUMOSEIEmotionDataset,
    RESDEmotionDataset,
    FIV2PersonalityDataset,
    BAHAbsencePresenceDataset,
)
from data.dataloaders import make_dataloader, make_concat_dataset


def _normalize_selection(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        items = []
        for a in x:
            if a is None:
                continue
            s = str(a).strip()
            if s:
                items.append(s)
        return items
    s = str(x).strip()
    if not s:
        return None
    if s.lower() == "all":
        return ["mosei", "resd", "fiv2", "bah"]
    return [p.strip() for p in s.split(",") if p.strip()]


def _emotion_mapping():
    return {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "happiness": 3,
        "neutral": 4,
        "sadness": 5,
        "enthusiasm": 6,
    }


def _selected_datasets(cfg):
    selected = _normalize_selection(getattr(cfg, "active_datasets", None))
    if selected is None:
        selected = _normalize_selection(getattr(cfg, "datasets", None))
    if selected is None:
        selected = ["mosei", "resd", "fiv2", "bah"]

    allowed = {"mosei", "resd", "fiv2", "bah"}
    selected = [s.lower() for s in selected]
    unknown = [s for s in selected if s not in allowed]
    if unknown:
        raise ValueError(f"Unknown datasets in selection: {unknown}. Allowed: {sorted(allowed)}")
    return selected


def _build_full_train_datasets(cfg, selected):
    emotion_mapping = _emotion_mapping()
    datasets = {}

    if "mosei" in selected:
        datasets["mosei"] = CMUMOSEIEmotionDataset(
            embeddings_npy_path=cfg.mosei_train_emb,
            labels_csv_path=cfg.mosei_train_csv,
        )

    if "resd" in selected:
        datasets["resd"] = RESDEmotionDataset(
            embeddings_npy_path=cfg.resd_train_emb,
            labels_csv_path=cfg.resd_train_csv,
            label_mapping=emotion_mapping,
        )

    if "fiv2" in selected:
        datasets["fiv2"] = FIV2PersonalityDataset(
            embeddings_npy_path=cfg.fiv2_train_emb,
            labels_csv_path=cfg.fiv2_csv,
            subset="train",
        )

    if "bah" in selected:
        datasets["bah"] = BAHAbsencePresenceDataset(
            embeddings_npy_path=cfg.bah_emb,
            labels_txt_path=cfg.bah_train_txt,
            sep=getattr(cfg, "bah_txt_sep", ","),
            id_field=getattr(cfg, "bah_txt_id_field", 0),
            label_field=getattr(cfg, "bah_txt_label_field", 1),
        )

    if not datasets:
        raise ValueError("No datasets selected for training.")
    return datasets


def _build_validation_loaders(cfg, selected):
    emotion_mapping = _emotion_mapping()
    val_loaders = {}

    if "mosei" in selected:
        val_loaders["mosei"] = make_dataloader(
            CMUMOSEIEmotionDataset(cfg.mosei_val_emb, cfg.mosei_val_csv),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "resd" in selected:
        val_loaders["resd"] = make_dataloader(
            RESDEmotionDataset(
                cfg.resd_test_emb,
                cfg.resd_test_csv,
                label_mapping=emotion_mapping,
            ),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "fiv2" in selected:
        val_loaders["fiv2"] = make_dataloader(
            FIV2PersonalityDataset(
                cfg.fiv2_val_emb,
                cfg.fiv2_csv,
                subset="validation",
            ),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "bah" in selected:
        val_loaders["bah"] = make_dataloader(
            BAHAbsencePresenceDataset(
                embeddings_npy_path=cfg.bah_emb,
                labels_txt_path=cfg.bah_val_txt,
                sep=getattr(cfg, "bah_txt_sep", ","),
                id_field=getattr(cfg, "bah_txt_id_field", 0),
                label_field=getattr(cfg, "bah_txt_label_field", 1),
            ),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    return val_loaders


def _build_test_loaders(cfg, selected):
    emotion_mapping = _emotion_mapping()
    test_loaders = {}

    if "mosei" in selected:
        test_loaders["mosei"] = make_dataloader(
            CMUMOSEIEmotionDataset(cfg.mosei_test_emb, cfg.mosei_test_csv),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "resd" in selected:
        test_loaders["resd"] = make_dataloader(
            RESDEmotionDataset(cfg.resd_test_emb, cfg.resd_test_csv, label_mapping=emotion_mapping),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "fiv2" in selected:
        test_loaders["fiv2"] = make_dataloader(
            FIV2PersonalityDataset(cfg.fiv2_test_emb, cfg.fiv2_csv, subset="test"),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    if "bah" in selected:
        test_loaders["bah"] = make_dataloader(
            BAHAbsencePresenceDataset(
                embeddings_npy_path=cfg.bah_emb,
                labels_txt_path=cfg.bah_test_txt,
                sep=getattr(cfg, "bah_txt_sep", ","),
                id_field=getattr(cfg, "bah_txt_id_field", 0),
                label_field=getattr(cfg, "bah_txt_label_field", 1),
            ),
            batch_size=cfg.eval_bs,
            shuffle=False,
        )

    return test_loaders


def build_experiment_loaders(cfg):
    selected = _selected_datasets(cfg)
    full_train = _build_full_train_datasets(cfg, selected)
    train_concat = make_concat_dataset([full_train[name] for name in selected])
    train_loader = make_dataloader(train_concat, batch_size=cfg.batch_size, shuffle=True)
    val_loaders = _build_validation_loaders(cfg, selected)
    test_loaders = _build_test_loaders(cfg, selected)
    return train_loader, val_loaders, test_loaders


def build_all_loaders(cfg):
    return build_experiment_loaders(cfg)
