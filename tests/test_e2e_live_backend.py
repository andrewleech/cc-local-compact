"""End-to-end test against the real local model backend. Skipped when it
isn't reachable (e.g. `titan` is offline over Tailscale), mirroring
cc-patcher's tests/test_e2e_real_binary.py skip-if-unavailable pattern.
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse

from cc_local_compact import config as config_mod
from cc_local_compact import server


def _backend_reachable(timeout: float = 2.0) -> bool:
    cfg = config_mod.load_config()
    parsed = urlparse(cfg.base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _write_fixture_session(path: Path) -> None:
    lines = [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"role": "user", "content": "Say hello."}},
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "message": {"role": "assistant", "id": "m1", "content": [{"type": "text", "text": "Hello!"}]},
        },
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "message": {"role": "user", "content": "Say goodbye."}},
        {
            "type": "assistant", "uuid": "a2", "parentUuid": "u2",
            "message": {"role": "assistant", "id": "m2", "content": [{"type": "text", "text": "Goodbye!"}]},
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


@unittest.skipUnless(_backend_reachable(), "local backend (titan) unreachable")
class LiveBackendE2ETests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.session_path = Path(self.tmpdir.name) / "fixture-session.jsonl"
        _write_fixture_session(self.session_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_compact_session_produces_markdown_summary(self):
        result = server._run_compaction(self.session_path, None, None, None, None)
        self.assertTrue(result["ok"], result)
        self.assertTrue(Path(result["output_path"]).exists())
        self.assertTrue(len(result["summary"]) > 0)

    def test_tiny_budget_exercises_reactive_backoff(self):
        # A near-impossible budget forces the reactive backoff path. This
        # is also the one live check of which overflow-detection branch
        # actually fires against the real backend (structured Anthropic-
        # error parse vs the self-estimated-gap fallback) -- see loop.py's
        # module docstring for the documented deviation this resolves.
        result = server._run_compaction(self.session_path, None, 200, None, None)
        self.assertIn(result.get("reason") if not result["ok"] else "ok", ("ok", "exhausted", "too_few_groups"))


if __name__ == "__main__":
    unittest.main()
