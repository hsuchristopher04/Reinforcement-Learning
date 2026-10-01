"""Check distributable files without executing models or downloading dependencies."""
from html.parser import HTMLParser
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git', '.venv', '__pycache__', 'runs'}


def release_files():
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_file() and path.suffix != '.pyc':
            yield path


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


def main():
    failures = []
    files = list(release_files())
    # Report filenames only, never print a potential credential.
    patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'gh[pousr]_[A-Za-z0-9]{36,}', r'github_pat_[A-Za-z0-9_]{60,}',
                r'AKIA[A-Z0-9]{16}']
    for path in files:
        if path.name.startswith('.env') or path.suffix in {'.pem', '.key'}:
            failures.append(f'Credential-like filename: {path.relative_to(ROOT)}')
        if path.suffix not in {'.py', '.md', '.html', '.json', '.txt', '.cjs', '.yml'}:
            continue
        text = path.read_text(encoding='utf-8-sig')
        if any(re.search(pattern, text) for pattern in patterns):
            failures.append(f'Potential credential: {path.relative_to(ROOT)}')
        if path.suffix == '.html' and path.name != 'demo_template.html':
            page = Page()
            page.feed(text)
            for link in page.links:
                if link.startswith(('#', 'https:', 'http:', 'data:', 'mailto:')):
                    continue
                target = (path.parent / link.split('#')[0]).resolve()
                if not target.is_relative_to(ROOT) or not target.is_file():
                    failures.append(f'Broken local link: {path.relative_to(ROOT)} -> {link}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {len(files)} distributable files; HTML local links and credential-pattern checks.')
    print('Pattern checks are not an exhaustive secret or security audit.')


if __name__ == '__main__':
    main()
