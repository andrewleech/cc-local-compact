import json

from cc_local_compact import register


EXECUTABLE = "/opt/cc-local-compact/bin/cc-local-compact"


def test_register_writes_command_file(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    command_path = home / "commands" / "remind.md"
    assert command_path.is_file()
    content = command_path.read_text()
    assert "description:" in content
    assert "argument-hint:" in content


def test_register_creates_settings_json_with_all_three_hooks(tmp_path):
    home = tmp_path / "claude_home"
    result = register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())

    upe_entries = settings["hooks"]["UserPromptExpansion"]
    assert len(upe_entries) == 1
    assert upe_entries[0]["matcher"] == "remind"
    assert len(upe_entries[0]["hooks"]) == 1
    remind_hook = upe_entries[0]["hooks"][0]
    assert remind_hook["command"] == EXECUTABLE
    assert remind_hook["args"] == ["remind-hook"]
    assert remind_hook["statusMessage"]  # non-empty (currently inert for this event type, see register.py)
    assert remind_hook["timeout"] > 600  # must exceed the 600s default -- a real compaction can take longer

    stop_entries = settings["hooks"]["Stop"]
    assert len(stop_entries) == 1
    assert "matcher" not in stop_entries[0]
    assert stop_entries[0]["hooks"] == [{"type": "command", "command": EXECUTABLE, "args": ["track-session"]}]

    session_start_entries = settings["hooks"]["SessionStart"]
    assert len(session_start_entries) == 1
    assert "matcher" not in session_start_entries[0]
    assert session_start_entries[0]["hooks"] == [{"type": "command", "command": EXECUTABLE, "args": ["track-session"]}]

    assert result["hooks_already_present"] == {"UserPromptExpansion": False, "Stop": False, "SessionStart": False}


def test_register_preserves_unrelated_existing_settings(tmp_path):
    home = tmp_path / "claude_home"
    home.mkdir()
    existing = {
        "someOtherSetting": True,
        "hooks": {
            "PreCompact": [{"matcher": "", "hooks": [{"type": "command", "command": "claude-net-mirror-push"}]}],
        },
    }
    (home / "settings.json").write_text(json.dumps(existing))

    register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())
    assert settings["someOtherSetting"] is True
    assert settings["hooks"]["PreCompact"] == existing["hooks"]["PreCompact"]
    assert settings["hooks"]["UserPromptExpansion"][0]["matcher"] == "remind"
    assert "Stop" in settings["hooks"]


def test_register_preserves_unrelated_existing_stop_hooks(tmp_path):
    home = tmp_path / "claude_home"
    home.mkdir()
    existing = {
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "some-other-stop-hook"}]}],
        },
    }
    (home / "settings.json").write_text(json.dumps(existing))

    register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())
    stop_hooks = settings["hooks"]["Stop"][0]["hooks"]
    assert {"type": "command", "command": "some-other-stop-hook"} in stop_hooks
    assert {"type": "command", "command": EXECUTABLE, "args": ["track-session"]} in stop_hooks


def test_register_is_idempotent(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)
    result = register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())
    assert len(settings["hooks"]["UserPromptExpansion"][0]["hooks"]) == 1
    assert len(settings["hooks"]["Stop"][0]["hooks"]) == 1
    assert len(settings["hooks"]["SessionStart"][0]["hooks"]) == 1
    assert result["hooks_already_present"] == {"UserPromptExpansion": True, "Stop": True, "SessionStart": True}


def test_register_refreshes_stale_status_message_on_already_installed_hook(tmp_path, monkeypatch):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    # simulate an older install whose statusMessage predates a HOOK_SPECS change
    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["hooks"]["UserPromptExpansion"][0]["hooks"][0]["statusMessage"] = "old stale message"
    settings_path.write_text(json.dumps(settings))

    result = register.register(home, executable=EXECUTABLE)

    settings = json.loads(settings_path.read_text())
    current_message = settings["hooks"]["UserPromptExpansion"][0]["hooks"][0]["statusMessage"]
    assert current_message != "old stale message"
    assert current_message == register.HOOK_SPECS[0]["statusMessage"]
    # still only one hook entry -- updated in place, not duplicated
    assert len(settings["hooks"]["UserPromptExpansion"][0]["hooks"]) == 1
    assert result["hooks_already_present"]["UserPromptExpansion"] is True


def test_register_refreshes_stale_timeout_on_already_installed_hook(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["hooks"]["UserPromptExpansion"][0]["hooks"][0]["timeout"] = 60  # simulate an older, too-short install
    settings_path.write_text(json.dumps(settings))

    register.register(home, executable=EXECUTABLE)

    settings = json.loads(settings_path.read_text())
    current_timeout = settings["hooks"]["UserPromptExpansion"][0]["hooks"][0]["timeout"]
    assert current_timeout == register.HOOK_SPECS[0]["timeout"]
    assert len(settings["hooks"]["UserPromptExpansion"][0]["hooks"]) == 1


def test_unregister_removes_command_file_and_all_hooks(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    result = register.unregister(home, executable=EXECUTABLE)

    assert result == {
        "ok": True, "command_file_removed": True,
        "hooks_removed": {"UserPromptExpansion": True, "Stop": True, "SessionStart": True},
    }
    assert not (home / "commands" / "remind.md").is_file()
    settings = json.loads((home / "settings.json").read_text())
    assert "UserPromptExpansion" not in settings.get("hooks", {})
    assert "Stop" not in settings.get("hooks", {})
    assert "SessionStart" not in settings.get("hooks", {})


def test_unregister_preserves_unrelated_hooks_and_settings(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)
    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["hooks"]["PreCompact"] = [{"matcher": "", "hooks": [{"type": "command", "command": "claude-net-mirror-push"}]}]
    settings["hooks"]["Stop"][0]["hooks"].append({"type": "command", "command": "some-other-stop-hook"})
    settings["someOtherSetting"] = True
    settings_path.write_text(json.dumps(settings))

    register.unregister(home, executable=EXECUTABLE)

    settings = json.loads(settings_path.read_text())
    assert settings["someOtherSetting"] is True
    assert settings["hooks"]["PreCompact"] == [{"matcher": "", "hooks": [{"type": "command", "command": "claude-net-mirror-push"}]}]
    assert settings["hooks"]["Stop"][0]["hooks"] == [{"type": "command", "command": "some-other-stop-hook"}]


def test_unregister_on_never_registered_is_a_safe_noop(tmp_path):
    home = tmp_path / "claude_home"
    result = register.unregister(home, executable=EXECUTABLE)
    assert result == {
        "ok": True, "command_file_removed": False,
        "hooks_removed": {"UserPromptExpansion": False, "Stop": False, "SessionStart": False},
    }
