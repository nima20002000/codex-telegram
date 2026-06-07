from __future__ import annotations

import re
from collections.abc import Callable

MDV2_PARSE_MODE = "MarkdownV2"

_MDV2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$")
_FENCE_MARKER_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def telegram_utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def escape_mdv2(text: str) -> str:
    return _MDV2_ESCAPE_RE.sub(r"\\\1", text)


def strip_mdv2(text: str) -> str:
    cleaned = re.sub(r"\\([_*\[\]()~`>#+\-=|{}.!\\])", r"\1", text)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"~([^~]+)~", r"\1", cleaned)
    cleaned = re.sub(r"\|\|([^|]+)\|\|", r"\1", cleaned)
    return cleaned


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _fence_marker(line: str) -> str | None:
    match = _FENCE_MARKER_RE.match(line)
    return match.group(1) if match else None


def _closes_fence(line: str, opener: str) -> bool:
    match = _FENCE_MARKER_RE.match(line)
    if match is None:
        return False
    marker = match.group(1)
    if marker[0] != opener[0] or len(marker) < len(opener):
        return False
    return line[match.end() :].strip() == ""


def _render_table_block(table_block: list[str]) -> str:
    if len(table_block) < 3:
        return "\n".join(table_block)
    headers = _split_table_row(table_block[0])
    if len(headers) < 2:
        return "\n".join(table_block)

    first_data_row = _split_table_row(table_block[2]) if len(table_block) > 2 else []
    has_row_label_col = len(first_data_row) == len(headers) + 1

    rendered_groups: list[str] = []
    for index, row in enumerate(table_block[2:], start=1):
        cells = _split_table_row(row)
        if has_row_label_col:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            data_cells = cells[1:]
        else:
            heading = next((cell for cell in cells if cell), f"Row {index}")
            data_cells = cells

        if len(data_cells) < len(headers):
            data_cells.extend([""] * (len(headers) - len(data_cells)))
        elif len(data_cells) > len(headers):
            data_cells = data_cells[: len(headers)]

        bullets = []
        for header, value in zip(headers, data_cells):
            if not has_row_label_col and value == heading:
                continue
            bullets.append(f"• {header}: {value}")
        rendered_groups.append("\n".join([f"**{heading}**", *bullets]))
    return "\n\n".join(rendered_groups)


def wrap_markdown_tables(text: str) -> str:
    if "|" not in text or "-" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    fence_marker: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        marker = _fence_marker(stripped)
        if marker is not None:
            if fence_marker is None:
                fence_marker = marker
            elif _closes_fence(stripped, fence_marker):
                fence_marker = None
            out.append(line)
            i += 1
            continue
        if fence_marker is not None:
            out.append(line)
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            table_block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and _is_table_row(lines[j]):
                table_block.append(lines[j])
                j += 1
            out.append(_render_table_block(table_block))
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _protect_fenced_blocks(text: str, placeholder: Callable[[str], str]) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        opener = _fence_marker(lines[i])
        if opener is None:
            out.append(lines[i])
            i += 1
            continue

        opening_line = lines[i]
        body_lines: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            if _closes_fence(lines[i], opener):
                closed = True
                i += 1
                break
            body_lines.append(lines[i])
            i += 1

        if not closed:
            out.append(opening_line)
            out.extend(body_lines)
            continue

        stripped_opening = opening_line.lstrip()
        info = stripped_opening[len(opener) :]
        body = "".join(body_lines).replace("\\", "\\\\").replace("`", "\\`")
        out.append(placeholder("```" + info + body + "```"))
    return "".join(out)


def format_message_mdv2(content: str) -> str:
    if not content:
        return content

    placeholders: dict[str, str] = {}
    counter = 0

    def ph(value: str) -> str:
        nonlocal counter
        key = f"\x00PH{counter}\x00"
        counter += 1
        placeholders[key] = value
        return key

    text = wrap_markdown_tables(content)
    text = _protect_fenced_blocks(text, ph)
    text = re.sub(r"(`[^`\n]+`)", lambda m: ph(m.group(0).replace("\\", "\\\\")), text)

    def convert_link(match: re.Match[str]) -> str:
        display = escape_mdv2(match.group(1))
        url = match.group(2).replace("\\", "\\\\").replace(")", "\\)")
        return ph(f"[{display}]({url})")

    text = re.sub(r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)", convert_link, text)

    def convert_header(match: re.Match[str]) -> str:
        inner = re.sub(r"\*\*(.+?)\*\*", r"\1", match.group(1).strip())
        return ph(f"*{escape_mdv2(inner)}*")

    text = re.sub(r"^#{1,6}\s+(.+)$", convert_header, text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: ph(f"*{escape_mdv2(m.group(1))}*"), text)
    text = re.sub(r"\*([^*\n]+)\*", lambda m: ph(f"_{escape_mdv2(m.group(1))}_"), text)
    text = re.sub(r"~~(.+?)~~", lambda m: ph(f"~{escape_mdv2(m.group(1))}~"), text)
    text = re.sub(r"\|\|(.+?)\|\|", lambda m: ph(f"||{escape_mdv2(m.group(1))}||"), text)

    def convert_blockquote(match: re.Match[str]) -> str:
        prefix = match.group(1)
        content = match.group(2)
        if prefix.startswith("**") and content.endswith("||"):
            return ph(f"{prefix} {escape_mdv2(content[:-2])}||")
        return ph(f"{prefix} {escape_mdv2(content)}")

    text = re.sub(r"^((?:\*\*)?>{1,3}) (.+)$", convert_blockquote, text, flags=re.MULTILINE)
    text = escape_mdv2(text)

    for key in reversed(list(placeholders.keys())):
        text = text.replace(key, placeholders[key])

    code_split = re.split(r"(```[\s\S]*?```|`[^`]+`)", text)
    safe_parts: list[str] = []
    for index, segment in enumerate(code_split):
        if index % 2 == 1:
            safe_parts.append(segment)
            continue

        def escape_bare(match: re.Match[str], current: str = segment) -> str:
            start = match.start()
            char = match.group(0)
            if start > 0 and current[start - 1] == "\\":
                return char
            if char == "(" and start > 0 and current[start - 1] == "]":
                return char
            if char == ")":
                before = current[:start]
                if "](" in before:
                    depth = 0
                    for pos in range(start - 1, max(start - 2000, -1), -1):
                        if current[pos] == "(":
                            depth -= 1
                            if depth < 0:
                                if pos > 0 and current[pos - 1] == "]":
                                    return char
                                break
                        elif current[pos] == ")":
                            depth += 1
            return "\\" + char

        safe_parts.append(re.sub(r"[(){}]", escape_bare, segment))
    return "".join(safe_parts)


def _open_code_fence_lang(text: str) -> str | None:
    in_fence = False
    lang = ""
    for match in re.finditer(r"(^|\n)```([^\n`]*)", text):
        if not in_fence:
            in_fence = True
            lang = match.group(2).strip()
        else:
            in_fence = False
            lang = ""
    return lang if in_fence else None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    pos = index - 1
    while pos >= 0 and text[pos] == "\\":
        backslashes += 1
        pos -= 1
    return backslashes % 2 == 1


def _inline_code_open_after(text: str, initially_open: bool) -> bool:
    in_fence = False
    in_inline = initially_open
    i = 0
    while i < len(text):
        if text.startswith("```", i) and not _is_escaped(text, i):
            in_fence = not in_fence
            i += 3
            continue
        if text[i] == "`" and not in_fence and not _is_escaped(text, i):
            in_inline = not in_inline
        i += 1
    return in_inline


def _slice_by_units(text: str, unit_limit: int) -> int:
    units = 0
    for index, char in enumerate(text):
        char_units = telegram_utf16_len(char)
        if units + char_units > unit_limit:
            return index
        units += char_units
    return len(text)


def split_telegram_text(text: str, *, limit: int = 4096) -> list[str]:
    if telegram_utf16_len(text) <= limit:
        return [text]

    indicator_reserve = 12
    chunks: list[str] = []
    remaining = text
    carry_lang: str | None = None
    carry_inline_code = False
    fence_close = "\n```"

    while remaining:
        if carry_lang is not None:
            prefix = f"```{carry_lang}\n"
        elif carry_inline_code:
            prefix = "`"
        else:
            prefix = ""
        headroom = limit - indicator_reserve - telegram_utf16_len(prefix) - telegram_utf16_len(fence_close)
        if headroom < 1:
            headroom = limit // 2
        if telegram_utf16_len(prefix + remaining) <= limit - indicator_reserve:
            final_chunk = prefix + remaining
            if carry_lang is None and _inline_code_open_after(remaining, carry_inline_code):
                final_chunk += "`"
            chunks.append(final_chunk)
            break

        slice_at = _slice_by_units(remaining, headroom)
        candidate = remaining[:slice_at]
        newline_at = candidate.rfind("\n")
        space_at = candidate.rfind(" ")
        split_at = max(newline_at, space_at)
        if split_at > max(1, len(candidate) // 2):
            candidate = candidate[: split_at + 1]
        if candidate.endswith("\\") and not _is_escaped(candidate, len(candidate) - 1):
            candidate = candidate[:-1]
        if not candidate:
            candidate = remaining[:slice_at]

        chunk = prefix + candidate
        carry_lang = _open_code_fence_lang(chunk)
        if carry_lang is not None:
            chunk += fence_close
            carry_inline_code = False
        else:
            carry_inline_code = _inline_code_open_after(candidate, carry_inline_code)
            if carry_inline_code:
                chunk += "`"
        chunks.append(chunk)
        remaining = remaining[len(candidate) :]

    total = len(chunks)
    if total <= 1:
        return chunks
    return [f"{chunk} \\({index}/{total}\\)" for index, chunk in enumerate(chunks, start=1)]
