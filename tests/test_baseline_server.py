import unittest
from pathlib import Path


class SubmissionContractTest(unittest.TestCase):
    def test_docker_submission_runs_the_l2_fastapi_app(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("COPY requirements.txt /app/requirements.txt", dockerfile)
        self.assertIn("COPY app.py /app/app.py", dockerfile)
        self.assertIn("COPY lunit_hackathon /app/lunit_hackathon", dockerfile)
        self.assertIn('"uvicorn", "app:app"', dockerfile)
        self.assertNotIn('"main.py", "serve"', dockerfile)
        self.assertTrue(dockerignore.startswith("**"))
        self.assertIn("!requirements.txt", dockerignore)
        self.assertIn("!app.py", dockerignore)
        self.assertIn("!lunit_hackathon/**", dockerignore)


if __name__ == "__main__":
    unittest.main()
