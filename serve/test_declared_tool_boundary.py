"""Fail-closed tool capabilities for upstream Strata 0.1.44.1."""
import unittest
from serve.frontend import OutputParser

DECLARED = [{"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
GOOD = "<tool_call>\n<function=write>\n<parameter=path>\nhello\n</parameter>\n</function>\n</tool_call>"
BAD = "<tool_call>\n<function=delete_everything>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>"

def events(text, tools, stream, step):
    p = OutputParser(thinking=False, tools=tools, stream_tools=stream)
    result = []
    for i in range(0, len(text), step):
        result.extend(p.feed(text[i:i+step]))
    result.extend(p.finish())
    return result

class DeclaredToolBoundary(unittest.TestCase):
    def test_undeclared_never_executable(self):
        for stream in (False, True):
            for step in (1, 7, 4096):
                with self.subTest(stream=stream, step=step):
                    found = events(BAD, DECLARED, stream, step)
                    self.assertFalse(any(e.kind.startswith("tool_") for e in found), found)
    def test_no_tools_no_executable_call(self):
        self.assertFalse(any(e.kind.startswith("tool_") for e in events(BAD, [], True, 1)))
    def test_truncated_undeclared_suppressed(self):
        for stream in (False, True):
            found = events(BAD[:-24], DECLARED, stream, 1)
            self.assertFalse(any(e.kind.startswith("tool_") for e in found), found)
    def test_declared_namespace_alias(self):
        declared = [{"name": "mcp__websearch.web_search", "parameters": {"type": "object",
                     "properties": {"query": {"type": "string"}}}}]
        output = "<tool_call>\n<function=mcp__websearch__web_search>\n<parameter=query>\nstrata\n</parameter>\n</function>\n</tool_call>"
        for stream in (False, True):
            found = events(output, declared, stream, 1)
            self.assertTrue(any(e.kind == "tool_call" and
                                e.call.name == "mcp__websearch__web_search" for e in found), found)
    def test_declared_preserved(self):
        for stream in (False, True):
            found = events(GOOD, DECLARED, stream, 1)
            self.assertTrue(any(e.kind == "tool_call" and e.call.name == "write" for e in found), found)

if __name__ == "__main__":
    unittest.main()
