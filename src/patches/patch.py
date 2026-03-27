from pathlib import Path
import textwrap


class Patch:
    def __init__(self, code_object: dict, optimized_code: str, root: str):
        self.code_object = code_object
        self.root = Path(root)
        self.file_path = self.root / self.code_object['rel_path']
        self._original_content: str | None = None

        # safety dedent
        optimized_code = textwrap.dedent(optimized_code)
        self.optimized_lines = optimized_code.splitlines(keepends=True)
        if self.optimized_lines and not self.optimized_lines[-1].endswith('\n'):
            self.optimized_lines[-1] += '\n'

    def apply_patch(self) -> bool:
        start = self.code_object['start_line']
        end = self.code_object['end_line']
        base_indent: int = self.code_object['base_indent']

        self._original_content = self.file_path.read_text(encoding='utf-8')
        old_lines = self._original_content.splitlines(keepends=True)

        # re-indent optimized code
        indent = ' ' * base_indent
        indented = [indent + line if line.strip() else line for line in self.optimized_lines]

        # check for no-op
        if indented == old_lines[start:end + 1]:
            self._original_content = None
            return False

        new_lines = old_lines[:start] + indented + old_lines[end + 1:]
        self.file_path.write_text(''.join(new_lines), encoding='utf-8')
        return True

    def revert_patch(self):
        if self._original_content is None:
            return
        self.file_path.write_text(self._original_content, encoding='utf-8')
        self._original_content = None


class PatchStack:
    """LIFO stack of applied patches for clean revert between agent runs."""

    def __init__(self):
        self._stack: list[Patch] = []

    def push(self, patch: Patch):
        self._stack.append(patch)

    def revert_all(self):
        while self._stack:
            self._stack.pop().revert_patch()