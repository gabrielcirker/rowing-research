"""Presentation dates stay out of the page chrome, not the report or archive."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1] / "docs"


class WebsiteChecks(unittest.TestCase):
    def test_presentation_omits_decorative_dates(self):
        text = (ROOT / "index.html").read_text()
        self.assertIn('<p class="author">Gabriel Cirker</p>', text)
        self.assertIn("Original manuscript: St. John’s University.", text)
        self.assertIn("Revision affiliation: University of Michigan, Ann Arbor.", text)
        self.assertNotIn("Gabriel Cirker · Revised", text)
        captions = re.findall(r"<figcaption>(Page .*?)</figcaption>", text)
        self.assertEqual(captions, [f"Page {n} of 12" for n in range(1, 13)])

    def test_report_and_archive_keep_necessary_dates(self):
        text = (ROOT / "index.html").read_text()
        self.assertIn("Original manuscript: December 2025; revised November 2026", text)
        self.assertIn("St. John’s University (original manuscript, 2025)", text)
        self.assertIn("University of Michigan, Ann Arbor (revision, 2026)", text)
        archive = (ROOT / "archive.html").read_text()
        self.assertIn("Current research revision (November 2026)", archive)


if __name__ == "__main__":
    unittest.main()
