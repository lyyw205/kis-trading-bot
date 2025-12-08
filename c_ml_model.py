# ml_model.py
# 모델의 로딩 관련 책임만 담당하는 초심플한 핵심 파일
# 지우면 안됨!
import os
import joblib

from c_db_manager import BotDatabase  # 로그용 (없어도 되지만 있으면 편함)

def load_model(path="models/model_v1_rf.pkl", db: BotDatabase | None = None):
    """
    ML 모델을 로드한다.
    - 파일이 없거나 에러나면 None 리턴 → 룰 기반으로만 동작
    """
    if not os.path.exists(path):
        msg = f"⚠️ ML 모델 파일 없음: {path} → 룰 기반으로만 진행합니다."
        if db:
            db.log(msg)
        else:
            print(msg)
        return None

    try:
        model = joblib.load(path)
        if db:
            db.log(f"✅ ML 모델 로드 완료: {path}")
        else:
            print(f"✅ ML 모델 로드 완료: {path}")
        return model
    except Exception as e:
        msg = f"❌ ML 모델 로드 실패: {e} → 룰 기반으로만 진행합니다."
        if db:
            db.log(msg)
        else:
            print(msg)
        return None
