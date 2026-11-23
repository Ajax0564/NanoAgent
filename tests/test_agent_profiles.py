import unittest

from nanoagent.profiles import AGENT_PROFILES
from nanoagent.toolbox import create_native_code_tools


PROFILE_TOOLS = {
    "math_agent": {"calculator", "plot_expression", "get_date_and_time"},
    "code_agent": {
        "list_files",
        "read_file",
        "read_lines",
        "find_files",
        "search_text",
        "inspect_file",
        "search_symbols",
        "write_file",
        "edit_file",
        "run_python",
        "run_python_file",
        "run_tests",
    },
    "file_agent": {"list_files", "read_file", "unzip_file"},
}
PROFILE_APPROVALS = {
    "math_agent": set(),
    "code_agent": {
        "write_file",
        "edit_file",
        "run_python",
        "run_python_file",
        "run_tests",
    },
    "file_agent": {"unzip_file"},
}


class AgentProfileTests(unittest.TestCase):
    def test_code_agent_guides_workspace_search_and_python_source_inspection(self):
        instructions = AGENT_PROFILES["code_agent"].instructions

        self.assertIn("create or update a file", instructions)
        self.assertIn("if they do not specify one", instructions)
        self.assertIn("*.py", instructions)
        self.assertIn("search_text", instructions)
        self.assertIn("configuration files", instructions)
        self.assertIn("inspect_file", instructions)
        self.assertIn("Never pass a directory", instructions)

    def test_profiles_expose_tools_from_only_their_configured_module(self):
        for name in AGENT_PROFILES:
            with self.subTest(profile=name):
                tools = create_native_code_tools(name)

                self.assertEqual(set(tools.registry), PROFILE_TOOLS[name])
                self.assertEqual(set(tools.requires_approval), PROFILE_APPROVALS[name])

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown agent profile"):
            create_native_code_tools("unknown_agent")


if __name__ == "__main__":
    unittest.main()
