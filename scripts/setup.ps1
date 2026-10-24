Write-Output "=== Starting AI Sign Language Bot Setup ==="

# Check Python version
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not installed or not in PATH."
    Exit 1
}

# Check Node version
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
node --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Node.js is not installed or not in PATH."
    Exit 1
}

Write-Output "Installing Python dependencies..."
pip install -r backend/requirements.txt

Write-Output "Installing Node.js frontend dependencies..."
Set-Location frontend
npm install
Set-Location ..

Write-Output "=== Setup Complete! You can now start the application using start_all.ps1 ==="
