@echo off
REM ================================
REM KIS 트레이딩 봇 - 매일 자동 재학습 스크립트
REM ================================

REM 1) 프로젝트 폴더로 이동
cd /d C:\Users\HP\Desktop\kis-trading-bot

REM 2) (선택) 가상환경 사용 시 활성화
REM call .venv\Scripts\activate

REM 3) ML 학습용 샘플 생성 (새로운 ohlcv + TP/SL 기반)
python build_ml_seq_samples.py

REM 4) 시퀀스 기반 ML 모델 학습 + 버전 저장 + active_model_path 갱신
python train_seq_model.py

echo --------------------------------------
echo 작업 완료! (build_ml_seq_samples + train_seq_model)
echo 로그는 trading.db 의 logs 테이블 / dashboard 에서 확인 가능
echo --------------------------------------
pause
