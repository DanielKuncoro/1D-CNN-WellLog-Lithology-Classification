import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR        = "C:\Daniel\TUGAS_AKHIR\IPCOAL\Model_CNN\Data_Well\TERBARU_YGBENER\FIX"
OUTPUT_DIR      = "C:\Daniel\TUGAS_AKHIR\IPCOAL\Model_CNN\MODEL_BARU_1\Hasil 5"
WINDOW_SIZE     = 101
FEATURES        = ["GR", "DENSITY"]
TARGET_COL      = "LITHO"
DEPTH_COL       = "DEPTH"
N_CLASSES       = 4
EPOCHS          = 100
BATCH_SIZE      = 64
MAX_CLASS_WEIGHT = 6.0

CLASS_NAMES = {0: "Claystone", 1: "Sandstone", 2: "Coaly Shale", 3: "Coal"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def focal_loss(gamma=2.0, alpha=1.0):
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        weight  = alpha * tf.pow(1.0 - y_pred, gamma)
        fl      = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return loss_fn


def load_wells(data_dir):
    wells = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".csv"):
            name = os.path.splitext(f)[0]
            df   = pd.read_csv(os.path.join(data_dir, f))
            df   = df.dropna(subset=FEATURES + [TARGET_COL, DEPTH_COL])
            df   = df.sort_values(DEPTH_COL).reset_index(drop=True)
            wells[name] = df
            print(f"  Loaded {name}: {len(df)} samples")
    return wells


def make_windows(df, scaler):
    X_scaled = scaler.transform(df[FEATURES].values)
    y_raw    = df[TARGET_COL].values.astype(int)
    half     = WINDOW_SIZE // 2
    X_win, y_win = [], []
    for i in range(half, len(X_scaled) - half):
        X_win.append(X_scaled[i - half: i + half + 1])
        y_win.append(y_raw[i])
    return np.array(X_win), np.array(y_win)


def prepare_data(wells, test_well_name):
    train_dfs = [df for name, df in wells.items() if name != test_well_name]
    scaler    = RobustScaler()
    scaler.fit(pd.concat(train_dfs)[FEATURES].values)

    train_X, train_y = [], []
    for name, df in wells.items():
        X_w, y_w = make_windows(df, scaler)
        if name != test_well_name:
            train_X.append(X_w)
            train_y.append(y_w)

    test_df        = wells[test_well_name]
    test_X, test_y = make_windows(test_df, scaler)

    return (np.concatenate(train_X), np.concatenate(train_y),
            test_X, test_y, scaler, test_df)


def get_capped_class_weights(y, max_weight=MAX_CLASS_WEIGHT):
    raw_cw = compute_class_weight("balanced",
                                   classes=np.arange(N_CLASSES),
                                   y=y)
    capped = np.clip(raw_cw, 1.0, max_weight)
    print(f"  Class weights (capped <= {max_weight}):")
    for i, (raw, cap) in enumerate(zip(raw_cw, capped)):
        print(f"    {CLASS_NAMES[i]:<14}: raw={raw:.2f}  -> capped={cap:.2f}")
    return dict(enumerate(capped))


def stratified_val_split(X, y, val_size=0.1):
    idx = np.arange(len(y))
    idx_train, idx_val = train_test_split(
        idx, test_size=val_size, random_state=SEED,
        stratify=y)
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]


def build_cnn(input_shape, n_classes):
    model = Sequential([
        Conv1D(64, kernel_size=7, activation="elu",
               padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, kernel_size=5, activation="elu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(256, kernel_size=3, activation="elu", padding="same"),
        BatchNormalization(),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation="elu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0),
        metrics=["accuracy"]
    )
    return model


def save_confusion_matrix(ytest, y_pred, well_name, acc):
    cm  = confusion_matrix(ytest, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
                yticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)])
    ax.set_title(f"Confusion Matrix — {well_name}  (Acc={acc:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"cm_{well_name}.png")
    plt.savefig(path, dpi=120)
    plt.close("all")
    print(f"  Saved confusion matrix  -> {path}")


def save_history(history, well_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    fig.suptitle(f"Training History — {well_name}", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"history_{well_name}.png")
    plt.savefig(path, dpi=120)
    plt.close("all")
    print(f"  Saved training history  -> {path}")


def run_lowo(data_dir=DATA_DIR):
    print("\nCNN COAL SEAM - LEAVE ONE WELL OUT (LOWO)")
    print("-" * 45)

    print("\nLoading wells...")
    wells      = load_wells(data_dir)
    well_names = list(wells.keys())
    assert len(well_names) == 9, \
        f"Dibutuhkan 9 sumur, ditemukan {len(well_names)}"

    all_results = []

    for fold, test_name in enumerate(well_names, 1):
        print(f"\nFold {fold}/9  |  Test Well: {test_name}")
        print("-" * 45)

        Xtrain_all, ytrain_all, Xtest, ytest, scaler, test_df = \
            prepare_data(wells, test_name)

        Xtrain, ytrain, Xval, yval = \
            stratified_val_split(Xtrain_all, ytrain_all, val_size=0.1)

        print(f"  Train: {len(ytrain)}  |  Val: {len(yval)}  |  Test: {len(ytest)}")

        cw_dict = get_capped_class_weights(ytrain)

        ytrain_cat = to_categorical(ytrain, N_CLASSES)
        yval_cat   = to_categorical(yval,   N_CLASSES)

        model = build_cnn((WINDOW_SIZE, len(FEATURES)), N_CLASSES)

        callbacks = [
            EarlyStopping(patience=7,
                          restore_best_weights=True,
                          monitor="val_loss",
                          mode="min",
                          verbose=1),
            ReduceLROnPlateau(factor=0.5,
                              patience=4,
                              monitor="val_loss",
                              mode="min",
                              min_lr=1e-5,
                              verbose=1)
        ]

        history = model.fit(
            Xtrain, ytrain_cat,
            validation_data=(Xval, yval_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=1
        )

        y_prob = model.predict(Xtest, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        acc    = accuracy_score(ytest, y_pred)

        print(f"\n  Accuracy blind well {test_name}: {acc:.4f}")
        print()
        print(classification_report(
            ytest, y_pred,
            target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)],
            zero_division=0))

        save_confusion_matrix(ytest, y_pred, test_name, acc)
        save_history(history, test_name)

        half        = WINDOW_SIZE // 2
        depth_valid = test_df[DEPTH_COL].values[half: len(test_df) - half]
        gr_valid    = test_df["GR"].values[half: len(test_df) - half]
        dens_valid  = test_df["DENSITY"].values[half: len(test_df) - half]

        df_pred = pd.DataFrame({
            "DEPTH":           np.round(depth_valid, 2),
            "GR":              gr_valid,
            "DENSITY":         dens_valid,
            "LITHO_ACTUAL":    ytest,
            "LITHO_PREDICTED": y_pred,
            "PROB_CLAYSTONE":  np.round(y_prob[:, 0], 4),
            "PROB_SANDSTONE":  np.round(y_prob[:, 1], 4),
            "PROB_COALYSHALE": np.round(y_prob[:, 2], 4),
            "PROB_COAL":       np.round(y_prob[:, 3], 4),
        })
        csv_path = os.path.join(OUTPUT_DIR, f"pred_{test_name}.csv")
        df_pred.to_csv(csv_path, index=False)
        print(f"  Saved prediction CSV    -> {csv_path}")

        all_results.append({
            "well":     test_name,
            "accuracy": round(acc, 4),
            "n_test":   len(ytest)
        })

        del model
        tf.keras.backend.clear_session()

    print("\nRingkasan LOWO")
    print("-" * 45)
    df_sum   = pd.DataFrame(all_results)
    mean_acc = df_sum["accuracy"].mean()
    std_acc  = df_sum["accuracy"].std()
    print(df_sum.to_string(index=False))
    print(f"\n  Mean Accuracy : {mean_acc:.4f}")
    print(f"  Std  Accuracy : {std_acc:.4f}")

    df_sum.to_csv(os.path.join(OUTPUT_DIR, "lowo_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(df_sum["well"], df_sum["accuracy"],
                  color="#4C72B0", edgecolor="white")
    ax.axhline(mean_acc, color="red", linestyle="--",
               label=f"Mean = {mean_acc:.3f} ± {std_acc:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("LOWO Accuracy per Blind Well — 1D-CNN")
    ax.legend()
    for bar, val in zip(bars, df_sum["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lowo_accuracy_summary.png"), dpi=130)
    plt.close("all")
    print(f"\n  Output folder: {os.path.abspath(OUTPUT_DIR)}")

    return df_sum


if __name__ == "__main__":
    summary = run_lowo(DATA_DIR)
