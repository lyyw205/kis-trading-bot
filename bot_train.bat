@echo off
cd /d C:\dev\stock-ml-project
echo [%DATE% %TIME%] Starting Morning Training Routine...

REM 가상환경 활성화
call .venv\Scripts\activate

REM 1. 데이터 갱신 (06:10 시작)
echo [1/3] Running build_ohlcv_history.py...
python build_ohlcv_history.py

REM 2. 샘플 생성 (앞 단계 끝나면 바로 실행)
echo [2/3] Running build_ml_seq_samples.py...
python build_ml_seq_samples.py

REM 3. 모델 학습 (앞 단계 끝나면 바로 실행)
echo [3/3] Running train_seq_model.py...
python train_seq_model.py

echo [%DATE% %TIME%] All Training Tasks Completed.
timeout /t 5
exit