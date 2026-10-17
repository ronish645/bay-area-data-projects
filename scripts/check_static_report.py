"""Verify that the deployed report's local links remain inside its artifact and exist."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "Bay_Area_Health_and_Pollution_Analysis" / "reports"


class LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attributes):
        for name, value in attributes:
            if value and ((tag == "a" and name == "href") or (tag == "img" and name == "src")):
                self.references.append(value)


def main():
    checked = 0
    for page in [ROOT / "index.html", ROOT / "legacy" / "index.html"]:
        parser = LocalReferences()
        parser.feed(page.read_text())
        for reference in parser.references:
            parts = urlsplit(reference)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.is_file():
                raise SystemExit(f"Missing or out-of-artifact reference in {page.name}: {reference}")
            checked += 1
    print(f"Static report verified: {checked} local file references resolve inside the Pages artifact.")


if __name__ == "__main__":
    main()
