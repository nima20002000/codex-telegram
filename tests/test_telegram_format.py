from __future__ import annotations

import unittest

from codex_telegram.telegram_format import (
    format_message_mdv2,
    split_telegram_text,
    strip_mdv2,
    telegram_utf16_len,
    wrap_markdown_tables,
)


class TelegramFormatTests(unittest.TestCase):
    def test_markdown_headings_and_inline_styles_convert_to_mdv2(self):
        rendered = format_message_mdv2("# Title\nThis is **bold** and *italic*.")

        self.assertIn("*Title*", rendered)
        self.assertNotIn("# Title", rendered)
        self.assertIn("*bold*", rendered)
        self.assertIn("_italic_", rendered)
        self.assertIn("\\.", rendered)

    def test_code_regions_are_preserved_and_escaped_for_mdv2(self):
        rendered = format_message_mdv2("Use `C:\\tmp\\file_name`.\n```sh\necho `date`\n```")

        self.assertIn("`C:\\\\tmp\\\\file_name`", rendered)
        self.assertIn("```sh\necho \\`date\\`\n```", rendered)

    def test_links_escape_label_without_corrupting_url(self):
        rendered = format_message_mdv2("[docs (v1)](https://example.test/a_(b)?q=1)")

        self.assertEqual(rendered, "[docs \\(v1\\)](https://example.test/a_\\(b\\)?q=1)")

    def test_tables_render_as_compact_row_groups(self):
        wrapped = wrap_markdown_tables(
            "Before\n"
            "| Service | State | Notes |\n"
            "|---|---|---|\n"
            "| API | Done | ok |\n"
            "| Web | Soon | needs CSS |\n"
            "\nAfter"
        )

        self.assertIn("**API**\n• State: Done\n• Notes: ok", wrapped)
        self.assertIn("**Web**\n• State: Soon\n• Notes: needs CSS", wrapped)
        self.assertNotIn("|---|", wrapped)

    def test_tables_inside_code_blocks_are_not_rewritten(self):
        text = "```text\n| A | B |\n|---|---|\n| 1 | 2 |\n```"

        self.assertEqual(wrap_markdown_tables(text), text)

    def test_tables_inside_tilde_code_blocks_are_not_rewritten(self):
        text = "~~~text\n| A | B |\n|---|---|\n| 1 | 2 |\n~~~"

        self.assertEqual(wrap_markdown_tables(text), text)

    def test_tilde_fence_with_inner_backtick_fence_keeps_tables_as_code(self):
        text = "~~~text\n```\n| A | B |\n|---|---|\n| 1 | 2 |\n```\n~~~"

        wrapped = wrap_markdown_tables(text)

        self.assertEqual(wrapped, text)
        self.assertNotIn("• B: 2", wrapped)

    def test_long_backtick_fence_keeps_inner_tables_as_code(self):
        text = "````text\n```\n| A | B |\n|---|---|\n| 1 | 2 |\n```\n````"

        wrapped = wrap_markdown_tables(text)
        rendered = format_message_mdv2(text)

        self.assertEqual(wrapped, text)
        self.assertNotIn("• B: 2", rendered)
        self.assertEqual(rendered, "```text\n\\`\\`\\`\n| A | B |\n|---|---|\n| 1 | 2 |\n\\`\\`\\`\n```")

    def test_nested_same_length_fence_with_info_string_stays_code(self):
        text = "```text\n```python\n| A | B |\n|---|---|\n| 1 | 2 |\n```\n```"

        wrapped = wrap_markdown_tables(text)
        rendered = format_message_mdv2(text)

        self.assertEqual(wrapped, text)
        self.assertNotIn("• B: 2", rendered)
        self.assertIn("\\`\\`\\`python", rendered)
        self.assertIn("|---|", rendered)

    def test_tilde_code_blocks_normalize_to_telegram_code_fences(self):
        rendered = format_message_mdv2("~~~text\n| A | B |\n|---|---|\n| 1 | 2 |\n~~~")

        self.assertEqual(rendered, "```text\n| A | B |\n|---|---|\n| 1 | 2 |\n```")
        self.assertNotIn("•", rendered)

    def test_split_uses_utf16_units_and_adds_part_indicators(self):
        self.assertEqual(telegram_utf16_len("a😀b"), 4)
        chunks = split_telegram_text("😀" * 30, limit=40)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].endswith("\\(1/3\\)"))
        self.assertTrue(chunks[-1].endswith("\\(3/3\\)"))
        self.assertTrue(all(telegram_utf16_len(chunk) <= 40 for chunk in chunks))

    def test_split_preserves_code_fences_across_chunks(self):
        chunks = split_telegram_text("```python\n" + ("print('x')\n" * 30) + "```", limit=120)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.count("```") >= 2 for chunk in chunks))
        self.assertTrue(chunks[0].startswith("```python\n"))
        self.assertTrue(chunks[1].startswith("```python\n"))

    def test_split_preserves_inline_code_across_chunks(self):
        rendered = format_message_mdv2("Result: `" + ("a" * 90) + "` done")

        chunks = split_telegram_text(rendered, limit=50)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.count("`") % 2 == 0 for chunk in chunks), chunks)
        self.assertTrue(chunks[0].startswith("Result: `"))
        self.assertTrue(chunks[1].startswith("`"))
        self.assertIn(" done", chunks[-1])

    def test_split_does_not_break_mdv2_escape_pairs(self):
        rendered = format_message_mdv2("a." * 20)

        chunks = split_telegram_text(rendered, limit=20)
        bodies = [chunk.rsplit(" \\(", 1)[0] for chunk in chunks]

        self.assertGreater(len(bodies), 1)
        self.assertEqual("".join(bodies), rendered)
        self.assertFalse(any(body.endswith("\\") for body in bodies), bodies)
        self.assertFalse(any(body.startswith(".") for body in bodies[1:]), bodies)

    def test_strip_mdv2_removes_formatting_fallback_noise(self):
        self.assertEqual(strip_mdv2("*Title*\nPath a\\.b \\(1/2\\)"), "Title\nPath a.b (1/2)")


if __name__ == "__main__":
    unittest.main()
