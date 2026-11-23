"""Data-access readiness tests using mocked HTTP and CLI responses only.

No real credentials are configured, no network download is performed and no
Kaggle terms are accepted in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import download_metadata as download_script
from sleeppsg_ci import data_access as da
from sleeppsg_ci.data_access import (
    CREDENTIAL_SOURCE_ACCESS_TOKEN_FILE,
    CREDENTIAL_SOURCE_API_TOKEN_ENV,
    CREDENTIAL_SOURCE_CONFIG_DIR,
    CREDENTIAL_SOURCE_ENV,
    CREDENTIAL_SOURCE_HOME,
    CREDENTIAL_SOURCE_NONE,
    CREDENTIAL_SOURCE_OAUTH,
    CREDENTIAL_SOURCE_XDG,
    SUPPORTED_CREDENTIAL_SOURCES,
    DataAccessError,
    access_audit_to_dict,
    audit_access,
    credential_source,
    download_metadata_files,
    fetch_remote_file_size,
    kaggle_credentials_present,
    validate_download_request,
    validate_remote_size,
)

SECRET = "SENTINEL_SECRET_SHOULD_NEVER_APPEAR_98765"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


def paginated(pages: list[dict]):
    calls: list[str] = []

    def opener(url: str, timeout: int = 30) -> FakeResponse:
        calls.append(url)
        return FakeResponse(pages[min(len(calls) - 1, len(pages) - 1)])

    return opener, calls


# --- credential detection ---------------------------------------------------

def test_environment_credentials(tmp_path):
    env = {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": SECRET}
    assert kaggle_credentials_present(environ=env, home=tmp_path / "empty")
    assert credential_source(environ=env, home=tmp_path / "empty") == CREDENTIAL_SOURCE_ENV


def test_home_kaggle_json_credentials(tmp_path):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "kaggle.json", {"username": "user", "key": SECRET})
    assert kaggle_credentials_present(environ={}, home=home)
    assert credential_source(environ={}, home=home) == CREDENTIAL_SOURCE_HOME


def test_custom_kaggle_config_dir(tmp_path):
    config = tmp_path / "custom_kaggle"
    write_json(config / "kaggle.json", {"username": "user", "key": SECRET})
    env = {"KAGGLE_CONFIG_DIR": str(config)}
    assert kaggle_credentials_present(environ=env, home=tmp_path / "empty")
    assert credential_source(environ=env, home=tmp_path / "empty") == CREDENTIAL_SOURCE_CONFIG_DIR


def test_api_token_environment_variable(tmp_path):
    env = {"KAGGLE_API_TOKEN": SECRET}
    assert kaggle_credentials_present(environ=env, home=tmp_path / "empty")
    assert credential_source(environ=env, home=tmp_path / "empty") == CREDENTIAL_SOURCE_API_TOKEN_ENV
    # Blank tokens are not credentials.
    assert not kaggle_credentials_present(environ={"KAGGLE_API_TOKEN": "   "}, home=tmp_path / "empty")


def test_api_token_env_pointing_to_token_file(tmp_path):
    token_file = tmp_path / "my_token.txt"
    token_file.write_text(SECRET + "\n", encoding="utf-8")
    env = {"KAGGLE_API_TOKEN": str(token_file)}
    assert credential_source(environ=env, home=tmp_path / "empty") == CREDENTIAL_SOURCE_API_TOKEN_ENV
    token_file.write_text("   \n", encoding="utf-8")
    assert not kaggle_credentials_present(environ=env, home=tmp_path / "empty")


def test_empty_api_token_file_skips_home_access_token(tmp_path):
    """An existing empty KAGGLE_API_TOKEN file consumes the token stage."""
    home = tmp_path / "home"
    token_file = home / ".kaggle" / "access_token"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(SECRET, encoding="utf-8")
    empty_token = tmp_path / "empty_token"
    empty_token.write_text("", encoding="utf-8")
    env = {"KAGGLE_API_TOKEN": str(empty_token)}
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_NONE
    assert not kaggle_credentials_present(environ=env, home=home)


def test_empty_api_token_file_continues_to_legacy_credentials(tmp_path):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "access_token").write_text(SECRET, encoding="utf-8")
    empty_token = tmp_path / "empty_token"
    empty_token.write_text("  \n", encoding="utf-8")
    env = {
        "KAGGLE_API_TOKEN": str(empty_token),
        "KAGGLE_USERNAME": "user",
        "KAGGLE_KEY": "legacy-key",
    }
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_ENV


def test_nonexistent_api_token_path_is_literal_token(tmp_path):
    env = {"KAGGLE_API_TOKEN": str(tmp_path / "does_not_exist")}
    assert credential_source(environ=env, home=tmp_path / "empty") == CREDENTIAL_SOURCE_API_TOKEN_ENV


def test_directory_as_api_token_path_is_not_a_credential(tmp_path):
    """Safety deviation: SDK could raise IsADirectoryError; we reject and continue."""
    token_dir = tmp_path / "token_dir"
    token_dir.mkdir()
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "access_token").write_text(SECRET, encoding="utf-8")
    env = {"KAGGLE_API_TOKEN": str(token_dir)}
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_NONE
    assert not kaggle_credentials_present(environ=env, home=home)

    env["KAGGLE_USERNAME"] = "user"
    env["KAGGLE_KEY"] = "legacy-key"
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_ENV


@pytest.mark.parametrize("filename", ["access_token", "access_token.txt"])
def test_access_token_file(tmp_path, filename):
    home = tmp_path / "home"
    path = home / ".kaggle" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SECRET, encoding="utf-8")
    assert credential_source(environ={}, home=home) == CREDENTIAL_SOURCE_ACCESS_TOKEN_FILE
    path.write_text("  ", encoding="utf-8")
    assert not kaggle_credentials_present(environ={}, home=home)


@pytest.mark.parametrize("filename", [
    "access_token.bak", "access_token.key", "token.txt", "access_token.json",
    "credentials.json.txt", "kaggle.json.txt",
])
def test_unsupported_token_file_extensions_are_not_credentials(tmp_path, filename):
    home = tmp_path / "home"
    path = home / ".kaggle" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SECRET, encoding="utf-8")
    assert credential_source(environ={}, home=home) == CREDENTIAL_SOURCE_NONE
    assert not kaggle_credentials_present(environ={}, home=home)


def test_oauth_credentials_file(tmp_path):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "credentials.json", {
        "refresh_token": SECRET, "access_token": "ignored", "username": "user",
    })
    assert credential_source(environ={}, home=home) == CREDENTIAL_SOURCE_OAUTH
    write_json(home / ".kaggle" / "credentials.json", {"access_token": "only"})
    assert not kaggle_credentials_present(environ={}, home=home)
    (home / ".kaggle" / "credentials.json").write_text("{bad", encoding="utf-8")
    assert not kaggle_credentials_present(environ={}, home=home)


def test_xdg_config_dir_legacy_json(tmp_path):
    xdg = tmp_path / "xdg"
    write_json(xdg / "kaggle" / "kaggle.json", {"username": "user", "key": SECRET})
    env = {"XDG_CONFIG_HOME": str(xdg)}
    assert credential_source(
        environ=env, home=tmp_path / "empty", platform_name="linux"
    ) == CREDENTIAL_SOURCE_XDG


def test_default_xdg_fallback_legacy_json(tmp_path):
    """Without XDG_CONFIG_HOME the CLI uses ~/.config/kaggle on Linux."""
    home = tmp_path / "home"
    write_json(home / ".config" / "kaggle" / "kaggle.json",
               {"username": "user", "key": SECRET})
    assert credential_source(
        environ={}, home=home, platform_name="linux"
    ) == CREDENTIAL_SOURCE_XDG


def test_xdg_ignored_when_home_kaggle_dir_exists(tmp_path):
    """CLI uses an existing ~/.kaggle even without a valid kaggle.json."""
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)  # exists but empty
    xdg = tmp_path / "xdg"
    write_json(xdg / "kaggle" / "kaggle.json", {"username": "user", "key": SECRET})
    assert credential_source(
        environ={"XDG_CONFIG_HOME": str(xdg)}, home=home, platform_name="linux"
    ) == CREDENTIAL_SOURCE_NONE


def test_xdg_ignored_when_custom_config_dir_set(tmp_path):
    home = tmp_path / "empty"
    custom = tmp_path / "custom"
    custom.mkdir()
    xdg = tmp_path / "xdg"
    write_json(xdg / "kaggle" / "kaggle.json", {"username": "user", "key": SECRET})
    env = {"KAGGLE_CONFIG_DIR": str(custom), "XDG_CONFIG_HOME": str(xdg)}
    assert credential_source(
        environ=env, home=home, platform_name="linux"
    ) == CREDENTIAL_SOURCE_NONE


def test_xdg_ignored_on_non_linux_platforms(tmp_path):
    xdg = tmp_path / "xdg"
    write_json(xdg / "kaggle" / "kaggle.json", {"username": "user", "key": SECRET})
    env = {"XDG_CONFIG_HOME": str(xdg)}
    assert credential_source(
        environ=env, home=tmp_path / "empty", platform_name="win32"
    ) == CREDENTIAL_SOURCE_NONE


def test_precedence_token_beats_legacy_and_oauth(tmp_path):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "access_token").write_text(SECRET, encoding="utf-8")
    write_json(home / ".kaggle" / "kaggle.json", {"username": "user", "key": "legacy"})
    write_json(home / ".kaggle" / "credentials.json", {"refresh_token": "oauth"})
    env = {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "legacy"}
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_ACCESS_TOKEN_FILE
    env["KAGGLE_API_TOKEN"] = "token"
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_API_TOKEN_ENV


def test_precedence_legacy_beats_oauth(tmp_path):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "kaggle.json", {"username": "user", "key": "legacy"})
    write_json(home / ".kaggle" / "credentials.json", {"refresh_token": "oauth"})
    assert credential_source(environ={}, home=home) == CREDENTIAL_SOURCE_HOME
    env = {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "legacy"}
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_ENV


def test_precedence_custom_config_dir_beats_home_json_and_xdg(tmp_path):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "kaggle.json", {"username": "home", "key": "home-key"})
    custom = tmp_path / "custom"
    write_json(custom / "kaggle.json", {"username": "custom", "key": "custom-key"})
    xdg = tmp_path / "xdg"
    write_json(xdg / "kaggle" / "kaggle.json", {"username": "xdg", "key": "xdg-key"})
    env = {"KAGGLE_CONFIG_DIR": str(custom), "XDG_CONFIG_HOME": str(xdg)}
    assert credential_source(environ=env, home=home) == CREDENTIAL_SOURCE_CONFIG_DIR


def test_missing_credentials(tmp_path):
    assert not kaggle_credentials_present(environ={}, home=tmp_path / "empty")
    assert credential_source(environ={}, home=tmp_path / "empty") == CREDENTIAL_SOURCE_NONE


@pytest.mark.parametrize("payload", [
    {"username": "user"},
    {"key": "key"},
    {"username": "", "key": SECRET},
    {"username": "user", "key": ""},
    ["not", "a", "dict"],
])
def test_incomplete_or_invalid_config(tmp_path, payload):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "kaggle.json", payload)
    assert not kaggle_credentials_present(environ={}, home=home)


def test_invalid_json_and_missing_file(tmp_path):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text("{not json", encoding="utf-8")
    assert not kaggle_credentials_present(environ={}, home=home)
    assert not kaggle_credentials_present(environ={}, home=tmp_path / "other")


def test_audit_reports_source_without_values(tmp_path):
    env = {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": SECRET}
    report = access_audit_to_dict(audit_access(environ=env, home=tmp_path / "empty"))
    assert report["credential_source"] == CREDENTIAL_SOURCE_ENV
    assert report["kaggle_credentials_present"] is True
    assert SECRET not in json.dumps(report)


def test_all_supported_sources_are_listed_and_never_leak(tmp_path):
    home = tmp_path / "home"
    write_json(home / ".kaggle" / "credentials.json", {"refresh_token": SECRET})
    report = access_audit_to_dict(audit_access(environ={}, home=home))
    assert report["credential_source"] == CREDENTIAL_SOURCE_OAUTH
    assert set(report["supported_credential_sources"]) == set(SUPPORTED_CREDENTIAL_SOURCES)
    assert SECRET not in json.dumps(report)


def test_token_env_never_leaks_in_download_error(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", SECRET)
    monkeypatch.setattr(da.shutil, "which", lambda _name: None)
    with pytest.raises(DataAccessError) as exc:
        download_metadata_files(tmp_path / "data", confirm_bsdp_terms=True)
    assert SECRET not in str(exc.value)


def test_no_credential_leakage_in_error_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", SECRET)
    monkeypatch.setattr(da.shutil, "which", lambda _name: None)

    # No explicit terms confirmation: error must not echo any credential.
    with pytest.raises(DataAccessError) as exc:
        download_metadata_files(tmp_path / "data", confirm_bsdp_terms=False)
    assert SECRET not in str(exc.value)

    # Credentials present but CLI missing: blocker must not echo any credential.
    with pytest.raises(DataAccessError) as exc:
        download_metadata_files(tmp_path / "data", confirm_bsdp_terms=True)
    assert SECRET not in str(exc.value)


# --- allowlist and size checks ----------------------------------------------

def test_download_allowlist_and_size_cap():
    validate_download_request(["demographics.csv"], 131_863)
    for files in (["ICD_codes_CI.csv"], ["recordings.edf"],
                  ["demographics.csv", "ICD_codes_CI.csv"]):
        with pytest.raises(DataAccessError):
            validate_download_request(files, 131_863)
    with pytest.raises(DataAccessError):
        validate_download_request(["demographics.csv"], da.MAX_METADATA_BYTES + 1)


def test_remote_size_validation():
    validate_remote_size(131_863)
    for size in (0, -1, da.MAX_METADATA_BYTES + 1):
        with pytest.raises(DataAccessError):
            validate_remote_size(size)


def test_fetch_remote_size_from_paginated_inventory():
    first = {"datasetFiles": [{"name": "ICD_codes_CI.csv", "totalBytes": 88_731}],
             "nextPageToken": "page-2"}
    second = {"datasetFiles": [{"name": "demographics.csv", "totalBytes": 131_863}]}
    opener, calls = paginated([first, second])
    assert fetch_remote_file_size(opener=opener) == 131_863
    assert len(calls) == 2
    assert "pageToken=page-2" in calls[1]


def test_fetch_remote_size_rejects_non_allowlisted_file():
    with pytest.raises(DataAccessError):
        fetch_remote_file_size("ICD_codes_CI.csv", opener=paginated([])[0])


def test_fetch_remote_size_missing_file():
    opener, _ = paginated([{"datasetFiles": []}])
    with pytest.raises(DataAccessError, match="not found"):
        fetch_remote_file_size(opener=opener)


# --- download safeguards (all mocked) --------------------------------------

def _authorised(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", SECRET)
    monkeypatch.setattr(da.shutil, "which", lambda _name: "/usr/bin/kaggle")


def test_download_checks_remote_size_before_invoking_cli(tmp_path, monkeypatch):
    _authorised(monkeypatch)
    monkeypatch.setattr(da, "fetch_remote_file_size", lambda *_a, **_k: da.MAX_METADATA_BYTES + 1)
    called = []

    def runner(*_a, **_k):
        called.append(True)
        raise AssertionError("CLI must not run when the remote file is too large")

    monkeypatch.setattr(da.subprocess, "run", runner)
    with pytest.raises(DataAccessError, match="above the"):
        download_metadata_files(tmp_path / "data", confirm_bsdp_terms=True)
    assert called == []


def test_download_requires_explicit_terms_confirmation(tmp_path, monkeypatch):
    _authorised(monkeypatch)
    monkeypatch.setattr(da, "fetch_remote_file_size", lambda *_a, **_k: pytest.fail("no fetch"))
    with pytest.raises(DataAccessError, match="explicitly confirmed"):
        download_metadata_files(tmp_path / "data", confirm_bsdp_terms=False)


def test_download_verifies_extracted_size(tmp_path, monkeypatch):
    _authorised(monkeypatch)
    monkeypatch.setattr(da, "fetch_remote_file_size", lambda *_a, **_k: 100)
    target = tmp_path / "data"
    target.mkdir()

    def runner(command, **_kwargs):
        (target / "demographics.csv").write_bytes(b"x" * 50)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(da.subprocess, "run", runner)
    with pytest.raises(DataAccessError, match="size mismatch"):
        download_metadata_files(target, confirm_bsdp_terms=True)


def test_download_success_with_mocked_cli(tmp_path, monkeypatch):
    _authorised(monkeypatch)
    monkeypatch.setattr(da, "fetch_remote_file_size", lambda *_a, **_k: 100)
    target = tmp_path / "data"

    def runner(command, **_kwargs):
        assert command[0] == "kaggle" and "demographics.csv" in command
        (target / "demographics.csv").write_bytes(b"x" * 100)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(da.subprocess, "run", runner)
    paths = download_metadata_files(target, confirm_bsdp_terms=True)
    assert paths == [target / "demographics.csv"]
    assert paths[0].stat().st_size == 100


# --- download script --------------------------------------------------------

def test_script_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(download_script, "fetch_remote_file_size", lambda *_a, **_k: pytest.fail("no fetch"))
    assert download_script.main(["--destination", str(tmp_path / "data")]) == 2


def test_script_without_credentials_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_script, "audit_access",
        lambda *_a, **_k: audit_access(environ={}, home=tmp_path / "empty"),
    )
    monkeypatch.setattr(download_script, "fetch_remote_file_size", lambda *_a, **_k: pytest.fail("no fetch"))
    code = download_script.main([
        "--confirm-bsdp-terms-accepted", "--destination", str(tmp_path / "data")
    ])
    assert code == 2


def test_script_parser_requires_explicit_flag():
    args = download_script.build_parser().parse_args([])
    assert args.confirm_bsdp_terms_accepted is False
    assert args.destination == "data"
