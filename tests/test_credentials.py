import json

from hexagon_kit.cli import main
from hexagon_kit.credentials import credentials_status, hf_token, qai_hub_token, save_tokens


def test_save_and_read_tokens(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv("HEXAGON_KIT_SECRETS", str(secrets))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HEXAGON_HF_TOKEN", raising=False)
    monkeypatch.delenv("QAI_HUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HEXAGON_QAI_HUB_TOKEN", raising=False)
    monkeypatch.setattr("hexagon_kit.credentials._hf_token_files", lambda: (tmp_path / "hf.token",))
    monkeypatch.setattr("hexagon_kit.credentials._qai_ini_path", lambda: tmp_path / "client.ini")

    assert hf_token() is None
    path = save_tokens(hf_token_value="hf_test_secret", qai_token_value="qai_test_secret")
    assert path == secrets
    stored = json.loads(secrets.read_text(encoding="utf-8"))
    assert stored["hf_token"] == "hf_test_secret"
    assert "hf_test_secret" not in json.dumps(credentials_status())
    assert credentials_status()["hf"] is True
    assert credentials_status()["qaiHub"] is True
    assert hf_token() == "hf_test_secret"
    assert qai_hub_token() == "qai_test_secret"
    assert (tmp_path / "hf.token").read_text(encoding="utf-8").strip() == "hf_test_secret"
    assert "qai_test_secret" in (tmp_path / "client.ini").read_text(encoding="utf-8")


def test_env_token_wins_over_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv("HEXAGON_KIT_SECRETS", str(secrets))
    secrets.write_text('{"hf_token": "from-file"}', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf_token() == "from-env"


def test_hub_configure_cli_does_not_print_secret(tmp_path, monkeypatch, capsys):
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv("HEXAGON_KIT_SECRETS", str(secrets))
    monkeypatch.setattr("hexagon_kit.credentials._hf_token_files", lambda: (tmp_path / "hf.token",))
    monkeypatch.setattr("hexagon_kit.credentials._qai_ini_path", lambda: tmp_path / "client.ini")
    assert main(["hub", "configure", "--hf-token", "hf_never_print_me"]) == 0
    out = capsys.readouterr().out
    assert "hf_never_print_me" not in out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["credentials"]["hf"] is True
