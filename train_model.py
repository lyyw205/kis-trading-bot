# train_model.py
import os
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DB_PATH = "trading.db"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_latest.pkl")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ml_samples", conn)
    conn.close()

    if df.empty:
        print("❌ ml_samples 테이블에 데이터가 없습니다. 먼저 build_ml_samples.py를 실행하세요.")
        raise SystemExit

    # 학습에 쓸 피처들 (실시간 트레이더가 쓰는 것과 맞추기)
    feature_cols = ["at_support", "is_bullish", "price_up", "lookback", "band_pct"]
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🔧 학습 샘플 수: {len(X_train)}, 검증 샘플 수: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 성능 리포트 출력
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 모델 저장
    joblib.dump(model, MODEL_PATH)
    print(f"💾 모델 저장 완료: {MODEL_PATH}")

