"""Unit tests for code-analysis PR comment permission handling."""

from contextlib import redirect_stderr
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import Mock, patch


class FakeGithubException(Exception):
    def __init__(self, status):
        self.status = status


class CodeAnalysisCommentTests(TestCase):
    @classmethod
    def setUpClass(cls):
        github = ModuleType("github")
        github.__path__ = []
        github.Github = Mock()
        exceptions = ModuleType("github.GithubException")
        exceptions.GithubException = FakeGithubException

        spec = spec_from_file_location("niftyreg_code_analysis", Path(__file__).with_name("code_analysis.py"))
        cls.analysis = module_from_spec(spec)
        with patch.dict("sys.modules", {"github": github, "github.GithubException": exceptions}):
            spec.loader.exec_module(cls.analysis)

    def setUp(self):
        self.pull = Mock()
        repo = Mock()
        repo.get_pull.return_value = self.pull
        client = Mock()
        client.get_repo.return_value = repo
        self.patcher = patch.object(self.analysis, "Github", return_value=client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_creates_comment_when_authorized(self):
        self.pull.get_issue_comments.return_value = []

        self.analysis.create_or_edit_comment("analysis result")

        self.pull.create_issue_comment.assert_called_once_with(body="analysis result")

    def test_edits_existing_comment_when_authorized(self):
        comment = Mock()
        comment.user.login = "github-actions[bot]"
        comment.body = self.analysis.COMMENT_TITLE + " previous result"
        self.pull.get_issue_comments.return_value = [comment]

        self.analysis.create_or_edit_comment("new result")

        comment.edit.assert_called_once_with(body="new result")
        self.pull.create_issue_comment.assert_not_called()

    def test_read_only_token_403_is_warning_and_preserves_report(self):
        self.pull.get_issue_comments.return_value = []
        self.pull.create_issue_comment.side_effect = FakeGithubException(403)
        stderr = StringIO()

        with redirect_stderr(stderr):
            self.analysis.create_or_edit_comment("analysis result")

        self.assertIn("::warning::", stderr.getvalue())
        self.assertIn("HTTP 403", stderr.getvalue())
        self.assertIn("analysis result", stderr.getvalue())

    def test_read_only_token_403_while_listing_is_warning(self):
        self.pull.get_issue_comments.side_effect = FakeGithubException(403)
        stderr = StringIO()

        with redirect_stderr(stderr):
            self.analysis.create_or_edit_comment("analysis result")

        self.assertIn("analysis result", stderr.getvalue())

    def test_other_api_errors_are_not_suppressed(self):
        self.pull.get_issue_comments.return_value = []
        self.pull.create_issue_comment.side_effect = FakeGithubException(500)

        with self.assertRaises(FakeGithubException):
            self.analysis.create_or_edit_comment("analysis result")


if __name__ == "__main__":
    main()
