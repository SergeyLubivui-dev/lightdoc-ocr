# Windows: python 3.11+ + uv (https://docs.astral.sh/uv/) или обычный venv
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv venv --python 3.11 .venv
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt
} else {
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
}
"OK: .venv готов. Запуск: run_docrec.bat <image> [out_dir]"
