Write-Output "=== Starting AI Sign Language Bot Servers ==="

# Load Node.js to path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Start FastAPI backend in a separate window/process
Write-Output "Launching FastAPI Backend on http://localhost:8000..."
Start-Process -FilePath "python" -ArgumentList "backend/start.py" -NoNewWindow:$false

# Start Next.js frontend in a separate window/process
Write-Output "Launching Next.js Frontend on http://localhost:3000..."
Set-Location frontend
Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -NoNewWindow:$false
Set-Location ..

Write-Output "=== Servers initiated! Please open http://localhost:3000 in your browser. ==="
