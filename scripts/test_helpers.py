import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.citations import convert_citations_to_markdown_links, extract_citations, validate_citations
from src.evaluation import normalize_ragas_scores


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class TestCitationHelpers(unittest.TestCase):
    def test_extract_and_convert_citations(self):
        text = "See [PR: #16021] and [Commit: 7fe315c21afb8a57a2b559772e0f7ced7e5d071a]."
        citations = extract_citations(text)

        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["type"], "pull_request")
        self.assertEqual(citations[1]["type"], "commit")

        retrieved_docs = [
            FakeDocument(
                page_content="Pull Request #16021: Optimize APIRoute handler caching",
                metadata={
                    "source_type": "pull_request",
                    "pr_number": 16021,
                    "source_url": "https://github.com/fastapi/fastapi/pull/16021",
                },
            ),
            FakeDocument(
                page_content="Commit SHA: 7fe315c21afb8a57a2b559772e0f7ced7e5d071a",
                metadata={
                    "source_type": "commit",
                    "commit_sha": "7fe315c21afb8a57a2b559772e0f7ced7e5d071a",
                    "source_url": "https://github.com/fastapi/fastapi/commit/7fe315c21afb8a57a2b559772e0f7ced7e5d071a",
                },
            ),
        ]

        validated, invalid = validate_citations(citations, retrieved_docs)
        self.assertEqual(invalid, [])

        linked = convert_citations_to_markdown_links(text, validated)
        self.assertIn("[PR #16021](https://github.com/fastapi/fastapi/pull/16021)", linked)
        self.assertIn(
            "[Commit #7fe315c21afb8a57a2b559772e0f7ced7e5d071a](https://github.com/fastapi/fastapi/commit/7fe315c21afb8a57a2b559772e0f7ced7e5d071a)",
            linked,
        )


class TestEvaluationHelpers(unittest.TestCase):
    def test_normalize_ragas_scores_from_dict_like_object(self):
        class Result:
            def __init__(self):
                self.scores = {"faithfulness": 0.8, "answer_relevancy": 0.7}

        scores = normalize_ragas_scores(Result())
        self.assertEqual(scores["faithfulness"], 0.8)
        self.assertEqual(scores["answer_relevancy"], 0.7)

    def test_normalize_ragas_scores_from_pandas_like_result(self):
        class FakeColumn:
            def __init__(self, values):
                self._values = values

            def mean(self):
                return sum(self._values) / len(self._values)

        class FakeDataFrame:
            def __init__(self, data):
                self._data = data
                self.columns = list(data.keys())
                self.empty = not bool(data)

            def select_dtypes(self, include=None):
                return self

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, key):
                return FakeColumn(self._data[key])

        class Result:
            def to_pandas(self):
                return FakeDataFrame(
                    {
                        "faithfulness": [0.8, 0.6],
                        "answer_relevancy": [0.7, 0.9],
                    }
                )

        scores = normalize_ragas_scores(Result())
        self.assertAlmostEqual(scores["faithfulness"], 0.7)
        self.assertAlmostEqual(scores["answer_relevancy"], 0.8)


if __name__ == "__main__":
    unittest.main()
