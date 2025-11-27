@echo off
cd /d C:\dev\stock-ml-project
echo [%DATE% %TIME%] Starting Trading Bot...

REM 가상환경 활성화
call .venv\Scripts\activate

REM 메인 실행
python main.py

pause