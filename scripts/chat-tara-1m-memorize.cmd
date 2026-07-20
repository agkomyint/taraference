@echo off
title Tara 1M Memorization Test
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0chat-tara-1m-memorize.ps1"
echo.
if errorlevel 1 echo The test failed. Copy the error above and send it to me.
pause
