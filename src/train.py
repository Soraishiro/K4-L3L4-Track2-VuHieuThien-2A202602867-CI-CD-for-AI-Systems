import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

F1_THRESHOLD = 0.65
BASE_POSITIVE_RATE = 0.248   # moc ty le lop duong cua bo du lieu goc
DRIFT_TOLERANCE = 0.05       # cho phep lech toi da 5% so voi moc


def setup_remote_tracking():
    """
    BONUS 1: theo doi MLflow tu xa tren DagsHub thay cho file sqlite cuc bo.

    Chi kich hoat khi bien moi truong DAGSHUB_TOKEN duoc dat. Neu khong co token
    thi MLflow tiep tuc dung backend sqlite cuc bo nhu Bước 1.
    """
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        print("[MLflow] Dung backend cuc bo (sqlite). Khong co DAGSHUB_TOKEN.")
        return None

    user = os.environ.get("DAGSHUB_USER", "thienmarco10")
    repo = os.environ.get("DAGSHUB_REPO", "mlflow")

    import dagshub
    import dagshub.auth

    dagshub.auth.add_app_token(token)
    # patch_mlflow=True: tuong thich MLflow 3.x voi API cu hon cua DagsHub,
    # tranh loi 404 o cac endpoint logged-models.
    dagshub.init(repo_name=repo, repo_owner=user, mlflow=True, patch_mlflow=True)

    mlflow.set_experiment("Income-Model")
    print(f"[MLflow] Tracking từ xa: https://dagshub.com/{user}/{repo}.mlflow")
    return mlflow.get_tracking_uri()


def scan_thresholds(y_true, proba, lo=0.1, hi=0.9, step=0.05):
    """
    BONUS 2: quet nguong xac suat 0.1 -> 0.9 de tim F1 toi uu thay vi mac dinh 0.5.

    Tra ve (best_threshold, best_f1, du_lieu_quet) trong do du_lieu_quet la
    danh sach dict de ghi vao MLflow va luu thanh artifact.
    """
    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 2)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "f1_score": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        })
    best = max(rows, key=lambda r: r["f1_score"])
    return best["threshold"], best["f1_score"], rows


def check_drift(y_true):
    """
    BONUS 5: canh bao lech lan khi ty le lop duong lech qua 5% so voi moc 24.8%.
    """
    rate = float(np.mean(y_true))
    rel = abs(rate - BASE_POSITIVE_RATE) / BASE_POSITIVE_RATE
    return {
        "positive_rate": rate,
        "baseline": BASE_POSITIVE_RATE,
        "relative_deviation": rel,
        "tolerance": DRIFT_TOLERANCE,
        "drift_detected": bool(rel > DRIFT_TOLERANCE),
    }


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    setup_remote_tracking()

    with mlflow.start_run():

        mlflow.log_params(params)

        # ---- BONUS 5: kiem tra lech lan du lieu ----
        drift = check_drift(y_train)
        if drift["drift_detected"]:
            print(
                "  [DRIFT CANH BAO] Ty le lop duong {:.2%} lech {:.1%} so voi moc {:.1%} "
                "> nguong {:.0%}".format(
                    drift["positive_rate"], drift["relative_deviation"],
                    drift["baseline"], drift["tolerance"],
                )
            )
        else:
            print(
                "  [DRIFT OK] Ty le lop duong {:.2%} trong nguong cho phep.".format(
                    drift["positive_rate"]
                )
            )

        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_eval)[:, 1]

        # Dung nguong mac dinh 0.5 de danh gia, dung lam chuan cho Quality Gate.
        preds = (proba >= 0.5).astype(int)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # ---- BONUS 3: precision / recall / confusion matrix ----
        precision = precision_score(y_eval, preds, zero_division=0)
        recall = recall_score(y_eval, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("true_positive", int(tp))
        mlflow.log_metric("false_positive", int(fp))
        mlflow.log_metric("true_negative", int(tn))
        mlflow.log_metric("false_negative", int(fn))

        # ---- BONUS 2: quet nguong toi uu ----
        best_t, best_f1, sweep = scan_thresholds(y_eval, proba)
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("f1_at_best_threshold", best_f1)
        mlflow.log_metric("positive_rate", drift["positive_rate"])
        mlflow.log_metric("drift_relative_deviation", drift["relative_deviation"])

        mlflow.sklearn.log_model(
            model, "model",
            serialization_format="pickle",
        )

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(
            "Confusion matrix: "
            f"[[tn={tn} fp={fp}] [fn={fn} tp={tp}]]"
        )
        print(
            f"Nguong toi uu: {best_t:.2f} (F1 toi uu = {best_f1:.4f}, "
            f"tang them {best_f1 - f1:+.4f} so voi nguong 0.5)"
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    # BONUS 3
                    "precision": precision,
                    "recall": recall,
                    "confusion_matrix": {
                        "tn": int(tn), "fp": int(fp),
                        "fn": int(fn), "tp": int(tp),
                    },
                    # BONUS 2
                    "best_threshold": best_t,
                    "f1_at_best_threshold": best_f1,
                    "threshold_sweep": sweep,
                    # BONUS 5
                    "drift": drift,
                },
                f,
                indent=2,
            )

        # Luu them 2 artifact phuc vu bao cao
        with open("outputs/precision_recall.json", "w") as f:
            json.dump(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "accuracy": acc,
                    "confusion_matrix": {
                        "tn": int(tn), "fp": int(fp),
                        "fn": int(fn), "tp": int(tp),
                    },
                },
                f,
                indent=2,
            )
        with open("outputs/drift_report.json", "w") as f:
            json.dump(drift, f, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)