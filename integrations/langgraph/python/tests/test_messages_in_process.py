import unittest
from unittest.mock import MagicMock


class TestMessagesInProcess(unittest.TestCase):
    def test_set_message_in_progress_tolerates_none_state(self):
        """Regression: messages_in_process must never crash when cleared."""
        from ag_ui_langgraph.agent import LangGraphAgent

        mock_graph = MagicMock()
        mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        mock_graph.config_schema.return_value = mock_config_schema

        agent = LangGraphAgent(name="test-agent", graph=mock_graph)

        run_id = "run-123"
        # Simulate legacy/previous behavior where clearing stored None
        agent.messages_in_process[run_id] = None

        agent.set_message_in_progress(
            run_id,
            {
                "id": "msg-1",
                "tool_call_id": "tool-1",
                "tool_call_name": "get_section_content",
            },
        )

        self.assertIsInstance(agent.messages_in_process[run_id], dict)
        self.assertEqual(agent.messages_in_process[run_id]["id"], "msg-1")


if __name__ == "__main__":
    unittest.main()

