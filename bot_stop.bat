@echo off
echo [%DATE% %TIME%] Stopping all Python processes...
taskkill /F /IM python.exe /T
echo Bot Stopped.
timeout /t 3
exit