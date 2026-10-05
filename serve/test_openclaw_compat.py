"""OpenClaw-focused compatibility regressions for Strata-5080 issue #4.

These tests intentionally avoid duplicating broad coverage already present in
serve/test_server.py and serve/test_responses.py.  They pin the additional
agent-safety boundary required by the fork: a model may only surface executable
tool calls whose names were declared by the caller.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


DECLARED = [{
    "name": "write",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    },
}]

GOOD = (
    "</think>\n\n"
    "<tool_call>\n"
    "<function=write>\n"
    "<parameter=path>\nnotes.txt\n</parameter>\n"
    "<parameter=content>\nhello\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

UNKNOWN = (
    "</think>\n\n"
    "<tool_call>\n"
    "<function=delete_everything>\n"
    "<parameter=path>\n/\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

UNKNOWN_CUT = (
    "</think>\n\n"
    "<tool_call>\n"
    "<function=delete_everything>\n"
    "<parameter=path>\n/important"
)


def parse(text: str, tools, stream_tools: bool, step: int):
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools)
    events = []
    for i in range(0, len(text), step):
        events += p.feed(text[i:i + step])
    events += p.finish()
    return events


class OpenClawRequestShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=16384)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                conn.request("POST", path, body=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "anthropic-version": "2023-06-01"})
                r = conn.getresponse()
                raw = r.read().decode()
            return r.status, json.loads(raw)
        finally:
            conn.close()

    def test_responses_accepts_benign_query_string(self):
        status, body = self.post("/v1/responses?beta=true", {
            "model": "m", "input": "reply with ok", "store": False,
            "reasoning": {"effort": "none"},
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(body["object"], "response")

    def test_empty_assistant_turn_keeps_later_messages_in_place(self):
        req = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": ""},
                {"role": "developer", "content": "Later reminder."},
                {"role": "user", "content": "second"},
            ]
        }
        messages, _tools, _kw = openai_to_messages(req)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user", "user"])
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(messages[3]["content"], "Later reminder.")

        status, body = self.post("/v1/chat/completions", {
            "model": "m", "max_tokens": 20, **req,
        })
        self.assertEqual(status, 200, body)


class DeclaredToolBoundary(unittest.TestCase):
    def test_declared_tool_is_unchanged(self):
        for stream in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream=stream, step=step):
                    events = parse(GOOD, DECLARED, stream, step)
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, "write")
                    self.assertEqual(calls[0].arguments, {"path": "notes.txt", "content": "hello"})

    def test_undeclared_whole_call_is_never_executable(self):
        for stream in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream=stream, step=step):
                    events = parse(UNKNOWN, DECLARED, stream, step)
                    self.assertFalse([e for e in events if e.kind in ("tool_start", "tool_args", "tool_call")])
                    self.assertFalse([e for e in events if "<tool_call>" in (e.text or "")])

    def test_undeclared_cut_call_is_not_streamed_or_leaked(self):
        for stream in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream=stream, step=step):
                    events = parse(UNKNOWN_CUT, DECLARED, stream, step)
                    self.assertFalse([e for e in events if e.kind in ("tool_start", "tool_args", "tool_call")])
                    self.assertFalse([e for e in events if "<tool_call>" in (e.text or "")])

    def test_no_tools_means_no_executable_tool_capability(self):
        for supplied in (None, []):
            for stream in (False, True):
                with self.subTest(supplied=supplied, stream=stream):
                    events = parse(UNKNOWN, supplied, stream, 3)
                    self.assertFalse([e for e in events if e.kind in ("tool_start", "tool_args", "tool_call")])

    def test_valid_first_call_survives_later_undeclared_truncation(self):
        text = GOOD + "\n\n" + UNKNOWN_CUT
        for stream in (False, True):
            for step in (1, 11, 10_000):
                with self.subTest(stream=stream, step=step):
                    events = parse(text, DECLARED, stream, step)
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual([(c.name, c.arguments) for c in calls],
                                     [("write", {"path": "notes.txt", "content": "hello"})])
                    self.assertFalse(any(
                        e.call is not None and e.call.name == "delete_everything"
                        for e in events
                    ))


if __name__ == "__main__":
    unittest.main()
