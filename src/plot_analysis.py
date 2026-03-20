def save_gallery_efficientnet_lowest_25():
    """
    Create and save gallery_efficientnet_preds_v2_lowest_25.png in the analysis_plots folder.
    """
    # Use project root as base (parent of src)
    base = Path(__file__).resolve().parent.parent
    outdir = base / 'outputs' / 'analysis_plots'
    preds_path = base / 'outputs' / 'preds' / 'efficientnet' / 'efficientnet_preds_v2.csv'
    test_csv = base / 'Data' / 'Annotations' / 'test.csv'
    spectrogram_base = base / 'Data' / 'Spectrograms'
    outdir.mkdir(parents=True, exist_ok=True)
    gallery(preds_path, test_csv, spectrogram_base, outdir, mode='lowest', n=25, overwrite=True)

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    import torch
except Exception:
    torch = None

# global default verbosity (can be overridden via run_analysis verbose arg)
VERBOSE = False


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def unique_path(p: Path) -> Path:
    """Return a Path that does not exist by appending an incrementing suffix.

    If `p` does not exist, it is returned unchanged. Otherwise returns
    `p` with suffix `_1`, `_2`, ... inserted before the file extension.
    """
    if not p.exists():
        return p
    parent = p.parent
    stem = p.stem
    suffix = p.suffix
    for i in range(1, 10000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find unique path for {p}")

def save_fig_if_not_exists(fig, out: Path, overwrite: bool = True, **save_kwargs):
    """Save a matplotlib figure to `out`.

    If `overwrite` is False and `out` exists, the function will close the figure
    and skip saving. If `overwrite` is True, it will overwrite the file.
    Returns True if a file was written, False otherwise.
    """
    # ensure parent dir exists
    out.parent.mkdir(parents=True, exist_ok=True)
    # Always save/overwrite the file (no skipping)
    try:
        fig.savefig(out, **save_kwargs)
        if VERBOSE:
            print(f'SAVED {out.name}')
        return True
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass

def plot_learning_curves(hist_paths, labels, outdir: Path, overwrite: bool = False, mode: str = 'side_by_side'):
    ensure_outdir(outdir)
    # Use a seaborn/matplotlib style that's available across versions.
    # Newer matplotlib may not accept 'seaborn-darkgrid', prefer a robust choice.
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        try:
            plt.style.use('seaborn-darkgrid')
        except Exception:
            plt.style.use('seaborn')
    if mode == 'side_by_side' and len(hist_paths) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for p, label in zip(hist_paths, labels):
            df = pd.read_csv(p)
            acc_train_col = 'train_acc' if 'train_acc' in df.columns else 'train_accuracy'
            acc_val_col = 'val_acc' if 'val_acc' in df.columns else 'val_accuracy'
            loss_train_col = 'train_loss'
            loss_val_col = 'val_loss'
            axes[0].plot(df['epoch'], df[acc_train_col], label=f'{label} train')
            axes[0].plot(df['epoch'], df[acc_val_col], '--', label=f'{label} val')
            axes[1].plot(df['epoch'], df[loss_train_col], label=f'{label} train')
            axes[1].plot(df['epoch'], df[loss_val_col], '--', label=f'{label} val')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('accuracy (%)')
        axes[0].legend()
        axes[0].set_title('Accuracy Learning Curve')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('loss')
        axes[1].legend()
        axes[1].set_title('Loss Learning Curve')
        fig.tight_layout()
        out = outdir / 'learning_curves_models.png'
        save_fig_if_not_exists(fig, out, overwrite=overwrite, dpi=150)
    else:
        # Plot each individually
        for p, label in zip(hist_paths, labels):
            df = pd.read_csv(p)
            acc_train_col = 'train_acc' if 'train_acc' in df.columns else 'train_accuracy'
            acc_val_col = 'val_acc' if 'val_acc' in df.columns else 'val_accuracy'
            loss_train_col = 'train_loss'
            loss_val_col = 'val_loss'
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(df['epoch'], df[acc_train_col], label='Train', marker='o')
            axes[0].plot(df['epoch'], df[acc_val_col], '--', label='Validation', marker='o')
            axes[0].set_xlabel('epoch')
            axes[0].set_ylabel('accuracy (%)')
            axes[0].legend()
            axes[0].set_title(f'{label} Accuracy Learning Curve')
            axes[1].plot(df['epoch'], df[loss_train_col], label='Train', marker='o')
            axes[1].plot(df['epoch'], df[loss_val_col], '--', label='Validation', marker='o')
            axes[1].set_xlabel('epoch')
            axes[1].set_ylabel('loss')
            axes[1].legend()
            axes[1].set_title(f'{label} Loss Learning Curve')
            fig.tight_layout()
            out = outdir / f'learning_curves_{label.lower()}.png'
            save_fig_if_not_exists(fig, out, overwrite=overwrite, dpi=150)


def plot_precision_recall_from_report(report_path: Path, outdir: Path, top_n: int = None, overwrite: bool = False):
    ensure_outdir(outdir)
    with open(report_path, 'r') as fh:
        report = json.load(fh)

    # filter out per-class metrics
    per_class = {k: v for k, v in report.items() if isinstance(v, dict)}
    df = pd.DataFrame(per_class).T
    df['precision'] = df['precision'].astype(float)
    df['recall'] = df['recall'].astype(float)
    df['support'] = df['support'].astype(float)

    if top_n is None:
        # show all classes
        df_sel = df.copy()
    else:
        # pick top_n by support as a candidate set
        df_sel = df.sort_values('support', ascending=False).head(top_n)

    n = len(df_sel)
    fig, ax = plt.subplots(1, 2, figsize=(14, max(6, n * 0.25 + 2)))

    # Order precision and recall independently, ascending (escalating)
    s_prec = df_sel['precision'].sort_values(ascending=True)
    s_rec = df_sel['recall'].sort_values(ascending=True)

    # Ensure the first two entries are 'weighted avg' then 'macro avg' when present
    def _promote(series, first_names=('weighted avg', 'macro avg')):
        # case-insensitive matching to preserve original index names
        idx_map = {i.lower(): i for i in series.index}
        promoted = []
        for n in first_names:
            key = n.lower()
            if key in idx_map:
                promoted.append(idx_map[key])
        if not promoted:
            return series
        rest = series.drop(promoted, errors='ignore')
        # build new series: promoted first in order, then rest (rest already sorted)
        values = [series.loc[p] for p in promoted] + rest.tolist()
        index = promoted + rest.index.tolist()
        return pd.Series(values, index=index)

    s_prec = _promote(s_prec)
    s_rec = _promote(s_rec)

    s_prec.plot.barh(ax=ax[0], color='C0')
    ax[0].set_title('Precision (ascending)')
    ax[0].set_xlabel('precision')

    s_rec.plot.barh(ax=ax[1], color='C1')
    ax[1].set_title('Recall (ascending)')
    ax[1].set_xlabel('recall')

    fig.tight_layout()
    out = outdir / (report_path.stem + '_precision_recall.png')
    save_fig_if_not_exists(fig, out, overwrite=overwrite, dpi=150)


def plot_strip_predictions(preds_path: Path, test_csv: Path, outdir: Path, top_n_species: int = None, max_per_species: int = 200, overwrite: bool = False):
    ensure_outdir(outdir)
    preds = pd.read_csv(preds_path)
    test = pd.read_csv(test_csv)
    if len(preds) == len(test):
        preds = preds.copy()
        preds['filename'] = test['filename'].values
    else:
        print('Warning: preds and test lengths differ; continuing without filenames')

    preds['correct'] = preds['y_true_name'] == preds['y_pred_name']

    # choose species by frequency in test set; show ascending (escalating) by example count
    counts = test['category'].value_counts()
    if top_n_species is None:
        # all species, descending by count (deescalating)
        top_species = counts.sort_values(ascending=False).index.tolist()
    else:
        # pick top_n by support then order them descending for display
        top_species = counts.sort_values(ascending=False).head(top_n_species).index.tolist()
    df_plot = preds[preds['y_true_name'].isin(top_species)].copy()

    # subsample per species to keep plot readable
    df_list = []
    for s in top_species:
        sub = df_plot[df_plot['y_true_name'] == s]
        if len(sub) > max_per_species:
            sub = sub.sample(max_per_species, random_state=42)
        df_list.append(sub)
    df_plot = pd.concat(df_list)

    # create explicit figure/axis so we save the right object
    fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(top_species) * 0.35)))

    if df_plot.empty:
        out = outdir / (preds_path.stem + '_stripplot.png')
        ax.text(0.5, 0.5, 'no data for strip plot', ha='center', va='center')
        ax.set_axis_off()
        fig.tight_layout()
        save_fig_if_not_exists(fig, out, dpi=150)
        return

    sns.stripplot(x='y_prob_max', y='y_true_name', data=df_plot, hue='correct', dodge=False, alpha=0.6, jitter=0.3, palette={True: 'C2', False: 'C3'}, ax=ax, order=top_species)
    ax.set_xlabel('predicted probability (max)')
    ax.set_ylabel('true species')
    ax.set_title('Strip plot of predictions by confidence (true vs false)')
    ax.legend(title='correct')
    out = outdir / (preds_path.stem + '_stripplot.png')
    fig.tight_layout()
    save_fig_if_not_exists(fig, out, overwrite=overwrite, dpi=150)


def load_spectrogram_tensor(path: Path):
    if not path.exists():
        return None
    if torch is None:
        print('torch not available; cannot load .pt spectrograms')
        return None
    try:
        t = torch.load(str(path), map_location='cpu')
    except Exception as e:
        print('error loading', path, e)
        return None

    # t might be a tensor or a dict with key 'spec' or 'mel' or similar
    if hasattr(t, 'numpy'):
        arr = t.numpy()
    elif isinstance(t, dict):
        # try common keys
        for k in ('spec', 'mel', 'spectrogram', 'S'):
            if k in t:
                v = t[k]
                arr = v.numpy() if hasattr(v, 'numpy') else np.array(v)
                break
        else:
            # pick first tensor-like value
            for v in t.values():
                if hasattr(v, 'numpy'):
                    arr = v.numpy()
                    break
            else:
                return None
    else:
        arr = np.array(t)

    # reduce dims
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        # fallback
        return None
    return arr


def gallery(preds_path: Path, test_csv: Path, spectrogram_base: Path, outdir: Path, mode: str = 'random', n: int = 25, overwrite: bool = False):
    ensure_outdir(outdir)
    preds = pd.read_csv(preds_path)
    test = pd.read_csv(test_csv)

    if len(preds) == len(test):
        preds = preds.copy()
        preds['filename'] = test['filename'].values
    else:
        print('Warning: preds and test lengths differ; filenames not attached')

    if mode == 'lowest':
        sel = preds.nsmallest(n, 'y_prob_max')
    else:
        # use truly random sampling each run (no fixed seed)
        sel = preds.sample(n)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis('off')

    # Build an index of available spectrogram files to avoid repeated filesystem searches.
    spec_index = {}
    try:
        for p in spectrogram_base.rglob('*.pt'):
            spec_index.setdefault(p.stem, []).append(p)
    except Exception:
        # spectrogram_base may not exist or rglob may fail; leave index empty
        spec_index = {}

    found_count = 0
    for i, (_, row) in enumerate(sel.iterrows()):
        ax = axes[i]
        filename = row.get('filename')
        true_name = row['y_true_name']
        pred_name = row['y_pred_name']
        prob = row['y_prob_max']
        # resolve filename stem (handle if filename already has .pt or path parts)
        if filename is None or (isinstance(filename, float) and np.isnan(filename)):
            filename = ''
        stem = Path(str(filename)).stem

        # 1) preferred location: spectrogram_base/test/<true_name>/<filename>.pt
        spec_path = spectrogram_base / 'test' / true_name / f'{stem}.pt'
        if not spec_path.exists():
            # 2) fallback: same folder but maybe filename already contained full name
            alt = spectrogram_base / 'test' / true_name / str(filename)
            if alt.exists():
                spec_path = alt
        if not spec_path.exists():
            # 3) lookup in indexed files by stem
            cand = spec_index.get(stem)
            if cand:
                spec_path = cand[0]
        if not spec_path.exists():
            # 4) try loose match on indexed stems
            found = None
            for k, lst in spec_index.items():
                if stem and (stem in k or k in stem):
                    found = lst[0]
                    break
            if found:
                spec_path = found

        arr = load_spectrogram_tensor(spec_path)
        if arr is None:
            ax.text(0.5, 0.5, 'no spectrogram', ha='center', va='center')
            ax.set_axis_off()
            continue



        # plot
        ax.imshow(arr, aspect='auto', origin='lower', cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(true_name, fontsize=8)
        caption = f'Pred: {pred_name} ({prob:.2f})'
        ax.text(0.5, -0.08, caption, transform=ax.transAxes, ha='center', fontsize=7)

    fig.suptitle(f'Gallery ({mode}) — {n} samples', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / f'gallery_{preds_path.stem}_{mode}_{n}.png'
    save_fig_if_not_exists(fig, out, overwrite=overwrite, dpi=150)
    # If running in a notebook, display the saved image instead of printing text
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=str(out)))
    except Exception:
        # fallback: do nothing
        pass


def _resolve_path(base: Path, p):
    """Resolve path `p` relative to `base` with a few fallbacks.

    Returns a Path object (may not exist).
    """
    p = Path(p) if p is not None else None
    if p is None:
        return None
    if p.exists():
        return p.resolve()
    # try relative to base
    try:
        cand = (Path(base) / p).resolve()
        if cand.exists():
            return cand
    except Exception:
        pass
    # try current working dir
    cw = Path.cwd()
    cand = (cw / p).resolve()
    if cand.exists():
        return cand
    # search by name under cwd
    matches = list(cw.glob('**/' + p.name))
    if matches:
        return matches[0].resolve()
    return p


try:
    from src.model import get_model
except Exception:
    try:
        from model import get_model
    except Exception:
        get_model = None


def run_inference_and_save(test_loader, classes, device, base: Path, model_name: str, ckpt_path: Path, overwrite_preds: bool = False):
    """Run inference using `test_loader` and save preds/misclassified files.

    `get_model` must be importable (from src.model or model).
    """
    if get_model is None:
        raise RuntimeError('get_model not available; cannot run inference')
    out_preds = Path(base) / 'outputs' / 'preds' / model_name / f'{model_name}_preds_v2.csv'
    out_mis = Path(base) / 'outputs' / 'misclassified' / model_name / f'{model_name}_misclassified_v2.csv'
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    out_mis.parent.mkdir(parents=True, exist_ok=True)

    # skip creating preds CSV unless overwrite_preds is True
    if out_preds.exists() and not overwrite_preds:
        if VERBOSE:
            print(f'SKIP inference {model_name}: preds exist ({out_preds.name})')
        return

    model = get_model(model_name, num_classes=(len(classes) if classes is not None else None), freeze_backbone=False)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_max_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1).cpu().numpy()
            y_true_all.extend(yb.numpy().tolist())
            y_pred_all.extend(preds.tolist())
            y_prob_max_all.extend(probs.max(dim=1).values.numpy().tolist())

    import pandas as pd
    df_pred = pd.DataFrame({'y_true': y_true_all, 'y_pred': y_pred_all, 'y_prob_max': y_prob_max_all})
    if classes is not None:
        df_pred['y_true_name'] = df_pred['y_true'].apply(lambda x: classes[x])
        df_pred['y_pred_name'] = df_pred['y_pred'].apply(lambda x: classes[x])

    df_pred.to_csv(out_preds, index=False)
    if VERBOSE:
        print(f'SAVED {out_preds.name}')

    mis = df_pred[df_pred['y_true'] != df_pred['y_pred']]
    mis.to_csv(out_mis, index=False)
    if VERBOSE:
        print(f'SAVED {out_mis.name}')


def run_inference_for_models(test_loader, classes, device, base: Path = Path('.'), model_names=None, overwrite_preds: bool = False):
    if model_names is None:
        model_names = ['efficientnet']
    for model_name in model_names:
        ckpt_name = f'{model_name}_best_2.pth'
        # try base first then glob
        p = Path(base) / ckpt_name
        if not p.exists():
            candidates = list(Path(base).glob(f'**/{ckpt_name.replace("_2","*")}'))
            if candidates:
                candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                p = candidates[0]
        if not p.exists():
            if VERBOSE:
                print(f'NO CKPT for {model_name} ({ckpt_name})')
            continue
        if VERBOSE:
            print('Using checkpoint:', p.name)
        run_inference_and_save(test_loader, classes, device, base, model_name, p, overwrite_preds=overwrite_preds)

def save_confusion_matrix(y_true, y_pred, class_names, out_dir="outputs/analysis_plots", filename="confusion_matrix.png"):
    """
    Plots and saves the confusion matrix for all classes.
    Args:
        y_true: List or array of true labels (integers or strings).
        y_pred: List or array of predicted labels (same type as y_true).
        class_names: List of all 44 class names, in order.
        out_dir: Directory to save the plot.
        filename: Name of the output image file.
    """
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(18, 18))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
    plt.title("Confusion Matrix (All 44 Classes)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def create_confusion_from_preds(preds_path: Path, test_csv: Path = None, class_names=None, out_dir: str = "outputs/analysis_plots", filename: str = "confusion_matrix_all_44.png"):
    """Convenience wrapper: load predictions file, infer labels, and save full confusion matrix.

    Args:
        preds_path: Path to predictions CSV. Expected columns: one of
            - 'y_true' and 'y_pred' (integer class indices),
            - 'y_true_name' and 'y_pred_name' (string class names), or
            - 'true_label' and 'predicted_label' (string names used in some notebooks).
        test_csv: optional Path to test.csv containing a 'category' column listing class names.
        class_names: optional list of class names (preferred). If omitted, will be inferred from
            preds or from `test_csv` if available.
        out_dir: output directory for saved image.
        filename: output filename.
    """
    preds_path = Path(preds_path)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    df = pd.read_csv(preds_path)

    # Determine whether preds contain numeric indices or names
    if 'y_true' in df.columns and 'y_pred' in df.columns:
        y_true = df['y_true'].astype(int).tolist()
        y_pred = df['y_pred'].astype(int).tolist()
        # need class_names to map indices -> labels for display
        if class_names is None:
            if test_csv is not None and Path(test_csv).exists():
                test_df = pd.read_csv(test_csv)
                if 'category' in test_df.columns:
                    class_names = sorted(test_df['category'].unique())
            else:
                raise ValueError('class_names required when preds use numeric indices')
    else:
        # try name-based columns
        if 'y_true_name' in df.columns and 'y_pred_name' in df.columns:
            y_true_names = df['y_true_name'].astype(str).tolist()
            y_pred_names = df['y_pred_name'].astype(str).tolist()
        elif 'true_label' in df.columns and 'predicted_label' in df.columns:
            y_true_names = df['true_label'].astype(str).tolist()
            y_pred_names = df['predicted_label'].astype(str).tolist()
        else:
            # fallback: try first two columns
            cols = list(df.columns)
            if len(cols) >= 2:
                y_true_names = df[cols[0]].astype(str).tolist()
                y_pred_names = df[cols[1]].astype(str).tolist()
            else:
                raise ValueError('Could not find suitable prediction columns in preds file')

        # derive class_names if not provided
        if class_names is None:
            if test_csv is not None and Path(test_csv).exists():
                test_df = pd.read_csv(test_csv)
                if 'category' in test_df.columns:
                    class_names = sorted(test_df['category'].unique())
            else:
                # use union of names found (sorted for deterministic order)
                class_names = sorted(set(y_true_names) | set(y_pred_names))

        # build mapping name -> index
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        # map names to indices, use -1 for unknown entries
        y_true = [name_to_idx.get(n, -1) for n in y_true_names]
        y_pred = [name_to_idx.get(n, -1) for n in y_pred_names]

    # filter out any unknown labels mapped to -1
    paired = [(t, p) for t, p in zip(y_true, y_pred) if t >= 0 and p >= 0]
    if not paired:
        raise ValueError('No valid prediction pairs found after mapping labels')
    y_true_f, y_pred_f = zip(*paired)

    # call the core saver
    save_confusion_matrix(list(y_true_f), list(y_pred_f), class_names, out_dir=out_dir, filename=filename)

def run_analysis(base: Path = Path('.'), history=None, report=None, preds=None, test_csv=None, spectrograms=None, out: str = 'outputs/analysis_plots', test_loader=None, classes=None, device='cpu', overwrite_plots: bool = True, overwrite_preds: bool = False, run_inference_flag: bool = False, gallery_mode: str = 'lowest', gallery_n: int = 25):
    """High-level function to run plotting + optional inference with simple defaults.

    - `base` is the repository root path.
    - If `run_inference_flag` is True, `test_loader` and `classes` must be provided.
    - `overwrite` controls whether generated plot files are overwritten.
    """
    base = Path(base).resolve()
    # prefer the legacy flat outputs/ files if present, fall back to organized subfolders
    hist_effnet_primary = base / 'outputs' / 'histories' / 'efficientnet_best_training_history_2.csv'
    HISTORY_EFFNET = _resolve_path(base, hist_effnet_primary)

    report_primary = base / 'outputs' / 'efficientnet_classification_report_v2.json'
    report_secondary = base / 'outputs' / 'reports' / 'efficientnet_classification_report_v2.json'
    REPORT = _resolve_path(base, report or report_primary)
    if (REPORT is None) or (not REPORT.exists()):
        REPORT = _resolve_path(base, report_secondary)

    preds_primary = base / 'outputs' / 'efficientnet_preds_v2.csv'
    preds_secondary = base / 'outputs' / 'preds' / 'efficientnet' / 'efficientnet_preds_v2.csv'
    PREDS = _resolve_path(base, preds or preds_primary)
    if (PREDS is None) or (not PREDS.exists()):
        PREDS = _resolve_path(base, preds_secondary)
    # default: Bens-Internship-Local is a sibling of the workspace root (two levels up from model folder)
    TEST_CSV = _resolve_path(base, test_csv or (base.parent.parent / 'Bens-Internship-Local' / 'Data' / 'Annotations' / 'test.csv'))
    SPECTROGRAMS = _resolve_path(base, spectrograms or (base / 'Data' / 'Spectrograms'))



    # Additional fallbacks: look for files under the workspace or sibling folders
    # Try sibling workspace 'Bens-Internship-Local' one level up from repository root
    try:
        if TEST_CSV is None or not TEST_CSV.exists():
            alt = (base.parent.parent / 'Bens-Internship-Local' / 'Data' / 'Annotations' / 'test.csv')
            if alt.exists():
                TEST_CSV = alt.resolve()
    except Exception:
        pass

    # Glob search for test.csv if still missing (search current working dir)
    if TEST_CSV is None or not TEST_CSV.exists():
        matches = list(Path.cwd().glob('**/test.csv'))
        for m in matches:
            # prefer ones under Bens-Internship-Local or Data/Annotations
            if 'Bens-Internship-Local' in str(m) or 'Annotations' in str(m):
                TEST_CSV = m.resolve()
                break
        if (TEST_CSV is None or not TEST_CSV.exists()) and matches:
            TEST_CSV = matches[0].resolve()

    # For spectrograms, prefer Bens-Internship-Local Data/Spectrograms if present
    try:
        if SPECTROGRAMS is None or not SPECTROGRAMS.exists():
            alt = base.parent.parent / 'Bens-Internship-Local' / 'Data' / 'Spectrograms'
            if alt.exists():
                SPECTROGRAMS = alt.resolve()
    except Exception:
        pass

    if SPECTROGRAMS is None or not SPECTROGRAMS.exists():
        # try searching cwd for a 'Spectrograms' folder
        matches = list(Path.cwd().glob('**/Spectrograms'))
        if matches:
            SPECTROGRAMS = matches[0].resolve()
    OUT = Path(base) / out
    OUT.mkdir(parents=True, exist_ok=True)

    # Only delete old files if explicitly requested (e.g., via a new argument)
    clean_analysis_plots = getattr(run_analysis, "clean_analysis_plots", False)
    if clean_analysis_plots:
        exts = ('*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf')
        for pat in exts:
            for f in OUT.glob(pat):
                try:
                    f.unlink()
                    if VERBOSE:
                        print(f'DELETE {f.name}')
                except Exception:
                    pass

    # optional inference
    if run_inference_flag:
        if test_loader is None:
            print('run_inference_flag set but no test_loader provided; skipping inference')
        else:
            run_inference_for_models(test_loader, classes, device, base=base, overwrite_preds=overwrite_preds)

    # plots
    # Plot EfficientNet learning curves if available
    hist_paths = []
    labels = []
    if HISTORY_EFFNET and HISTORY_EFFNET.exists():
        hist_paths.append(HISTORY_EFFNET)
        labels.append('EfficientNet')
    if hist_paths:
        # Plot side by side if both, and also individually
        if len(hist_paths) > 1:
            plot_learning_curves(hist_paths, labels, OUT, overwrite=overwrite_plots, mode='side_by_side')
            plot_learning_curves(hist_paths, labels, OUT, overwrite=overwrite_plots, mode='individual')
        else:
            plot_learning_curves(hist_paths, labels, OUT, overwrite=overwrite_plots, mode='individual')
    else:
        print('No history files found for learning curves.')

    if REPORT and REPORT.exists():
        plot_precision_recall_from_report(REPORT, OUT, overwrite=overwrite_plots)
    else:
        print('Report file not found:', REPORT)

    if PREDS and PREDS.exists() and TEST_CSV and TEST_CSV.exists():
        plot_strip_predictions(PREDS, TEST_CSV, OUT, overwrite=overwrite_plots)
    else:
        print('Preds or test CSV not found:', PREDS, TEST_CSV)

    if PREDS and PREDS.exists() and TEST_CSV and TEST_CSV.exists() and SPECTROGRAMS and SPECTROGRAMS.exists():
        gallery(PREDS, TEST_CSV, SPECTROGRAMS, OUT, mode=gallery_mode, n=gallery_n, overwrite=overwrite_plots)
    else:
        print('Gallery skipped; missing preds/test/spectrograms')


def main():
        # To create gallery_efficientnet_preds_v2_lowest_25.png in analysis_plots folder, uncomment below:
        # save_gallery_efficientnet_lowest_25()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='.', help='base path to model folder')
    parser.add_argument('--preds', type=str, default='outputs/efficientnet_preds_v2.csv')
    parser.add_argument('--test', type=str, default='../Bens-Internship-Local/Data/Annotations/test.csv')
    parser.add_argument('--history', type=str, default='outputs/efficientnet_best_training_history_2.csv')
    parser.add_argument('--report', type=str, default='outputs/efficientnet_classification_report_v2.json')
    parser.add_argument('--spectrograms', type=str, default='Data/Spectrograms')
    parser.add_argument('--out', type=str, default='outputs/analysis_plots')
    parser.add_argument('--gallery_mode', choices=['random', 'lowest'], default='random')
    parser.add_argument('--gallery_n', type=int, default=25)
    args = parser.parse_args()

    base = Path(args.base)
    outdir = base / args.out
    # Clear out all files in analysis_plots at the start of the run
    outdir.mkdir(parents=True, exist_ok=True)
    exts = ('*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf')
    for pat in exts:
        for f in outdir.glob(pat):
            try:
                f.unlink()
                if VERBOSE:
                    print(f'DELETE {f.name}')
            except Exception:
                pass

    hist_path = base / args.history
    report_path = base / args.report
    preds_path = base / args.preds
    test_csv = Path(args.test)
    spectrogram_base = base / args.spectrograms

    # learning curves
    if hist_path.exists():
        plot_learning_curves([hist_path], ['EfficientNet'], outdir)
    else:
        print('history file not found:', hist_path)

    # precision / recall
    if report_path.exists():
        plot_precision_recall_from_report(report_path, outdir)
    else:
        print('report file not found:', report_path)

    # strip plot
    if preds_path.exists() and test_csv.exists():
        plot_strip_predictions(preds_path, test_csv, outdir)
    else:
        print('preds or test csv not found')

    # gallery (random)
    if preds_path.exists() and test_csv.exists():
        gallery(preds_path, test_csv, spectrogram_base, outdir, mode='random', n=args.gallery_n)
        # gallery (lowest)
        gallery(preds_path, test_csv, spectrogram_base, outdir, mode='lowest', n=args.gallery_n)
    else:
        print('gallery skipped: preds/test missing')


if __name__ == '__main__':
    main()


