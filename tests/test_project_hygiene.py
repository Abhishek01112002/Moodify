from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_secret_files_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore
    assert "Auth.env" in gitignore
    assert "!.env.example" in gitignore


def test_readme_uses_portfolio_language():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Moodify" in readme
    assert "FAANG" not in readme
    assert "Evaluation" in readme
