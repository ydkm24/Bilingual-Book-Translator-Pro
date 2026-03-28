# Bilingual Book Translator Pro - Web UI Launcher
# Kill any existing processes
Get-Process | Where-Object { $_.Path -match "python" -or $_.Name -match "py" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Navigate to the new app directory and launch
cd "new app"
py app.py
