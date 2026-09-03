import json

from cc_local_compact import register


EXECUTABLE = "/opt/cc-local-compact/bin/cc-local-compact"


def test_register_writes_command_file(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    command_path = home / "commands" / "remind.md"
    assert command_path.is_file()
    assert "description:" in command_path.read_text()


def test_register_creates_settings_json_with_hook(tmp_path):
    home = tmp_path / "claude_home"
    result = register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())
    entries = settings["hooks"]["UserPromptExpansion"]
    assert len(entries) == 1
    assert entries[0]["matcher"] == "remind"
    assert entries[0]["hooks"] == [{"type": "command", "command": f"{EXECUTABLE} remind-hook"}]
    assert result["hook_already_present"] is False


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


def test_register_is_idempotent(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)
    result = register.register(home, executable=EXECUTABLE)

    settings = json.loads((home / "settings.json").read_text())
    assert len(settings["hooks"]["UserPromptExpansion"]) == 1
    assert len(settings["hooks"]["UserPromptExpansion"][0]["hooks"]) == 1
    assert result["hook_already_present"] is True


def test_unregister_removes_command_file_and_hook(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)

    result = register.unregister(home, executable=EXECUTABLE)

    assert result == {"ok": True, "command_file_removed": True, "hook_removed": True}
    assert not (home / "commands" / "remind.md").is_file()
    settings = json.loads((home / "settings.json").read_text())
    assert "UserPromptExpansion" not in settings.get("hooks", {})


def test_unregister_preserves_unrelated_hooks_and_settings(tmp_path):
    home = tmp_path / "claude_home"
    register.register(home, executable=EXECUTABLE)
    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["hooks"]["PreCompact"] = [{"matcher": "", "hooks": [{"type": "command", "command": "claude-net-mirror-push"}]}]
    settings["someOtherSetting"] = True
    settings_path.write_text(json.dumps(settings))

    register.unregister(home, executable=EXECUTABLE)

    settings = json.loads(settings_path.read_text())
    assert settings["someOtherSetting"] is True
    assert settings["hooks"]["PreCompact"] == [{"matcher": "", "hooks": [{"type": "command", "command": "claude-net-mirror-push"}]}]


def test_unregister_on_never_registered_is_a_safe_noop(tmp_path):
    home = tmp_path / "claude_home"
    result = register.unregister(home, executable=EXECUTABLE)
    assert result == {"ok": True, "command_file_removed": False, "hook_removed": False}
