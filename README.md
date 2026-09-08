<!-- README.md -->

# AI Data Analyst Application

Typecheck: `pyright` (from the repo root; checks `src/` only).
Tests: `pytest`.

## Run

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set the four `OLLAMA_MODEL_*` names in `.env` (placeholders in `.env.example`). Then:

```powershell
streamlit run src/ui/app.py
```
