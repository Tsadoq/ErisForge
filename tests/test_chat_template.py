import unittest
from unittest.mock import MagicMock

from erisforge.eris_forge import Forge


class TestChatTemplate(unittest.TestCase):
    def test_custom_template_is_used_when_tokenizer_has_none(self):
        custom_template = "{{ messages[0]['content'] }}"
        forge = Forge(custom_chat_template=custom_template)
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokens = MagicMock()
        tokenizer.apply_chat_template.return_value = tokens
        forge.load_instructions(["hello"], ["goodbye"])

        result = forge.tokenize_instructions(tokenizer, disable_tqdm=True)

        self.assertIs(result["objective_behaviour_tokens"][0], tokens)
        tokenizer.apply_chat_template.assert_any_call(
            conversation=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=custom_template,
        )
        tokenizer.apply_chat_template.assert_any_call(
            conversation=[{"role": "user", "content": "goodbye"}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=custom_template,
        )
        tokenizer.assert_not_called()

    def test_native_template_takes_precedence(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "native template"

        Forge._tokenize(tokenizer, "hello", custom_chat_template="custom template")

        tokenizer.apply_chat_template.assert_called_once_with(
            conversation=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        tokenizer.assert_not_called()

    def test_missing_template_keeps_plain_text_fallback(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.return_value = "plain text tokens"

        result = Forge._tokenize(tokenizer, "hello")

        self.assertEqual(result, "plain text tokens")
        tokenizer.assert_called_once_with(
            "User query: hello\nAI Assistant: ",
            return_tensors="pt",
        )
        tokenizer.apply_chat_template.assert_not_called()

    def test_invalid_custom_template_error_is_not_swallowed(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.apply_chat_template.side_effect = ValueError("invalid template")

        with self.assertRaisesRegex(ValueError, "invalid template"):
            Forge._tokenize(tokenizer, "hello", custom_chat_template="{{ invalid }")

        tokenizer.assert_not_called()

    def test_native_template_error_is_not_swallowed(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "native template"
        tokenizer.apply_chat_template.side_effect = ValueError("invalid native template")

        with self.assertRaisesRegex(ValueError, "invalid native template"):
            Forge._tokenize(tokenizer, "hello")

        tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
