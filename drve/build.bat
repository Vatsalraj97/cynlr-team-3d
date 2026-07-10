@echo off
echo [DRVE] Installing build deps...
pip install pyinstaller aiohttp numpy --quiet

echo [DRVE] Building standalone executable...
pyinstaller ^
  --onefile ^
  --noconsole ^
  --add-data "static;static" ^
  --add-data "engine.py;." ^
  --add-data "governor.py;." ^
  --add-data "data.py;." ^
  --add-data "events.py;." ^
  --add-data "feedback.py;." ^
  --add-data "server.py;." ^
  --name DRVE ^
  launch.py

echo.
echo [DRVE] Done. Executable at: dist\DRVE.exe
pause
