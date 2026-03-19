"""Tests for QY Domain resource server scoring logic."""

import pytest

from app import (
    score_connections,
    score_typos,
    score_unscrambling,
)


class TestTypos:
    def test_correct_answer_in_solution_tags(self):
        gt = "hello world"
        answer = "Here is the fix: <solution>hello world</solution>"
        assert score_typos(gt, answer) == 1.0

    def test_incorrect_answer(self):
        gt = "hello world"
        answer = "<solution>goodbye world</solution>"
        assert score_typos(gt, answer) == 0.0

    def test_no_solution_tags_correct(self):
        gt = "hello world"
        answer = "The corrected text is: hello world"
        assert score_typos(gt, answer) == 1.0

    def test_empty_answer(self):
        gt = "hello world"
        assert score_typos(gt, "") == 0.0


class TestConnections:
    def test_perfect_score(self):
        gt = "a,b,c,d,e,f,g,h"
        answer = "<solution>a,b,c,d,e,f,g,h</solution>"
        assert score_connections(gt, answer) == 1.0

    def test_zero_score(self):
        gt = "a,b,c,d,e,f,g,h"
        answer = "<solution>a,b,e,f,c,d,g,h</solution>"
        assert score_connections(gt, answer) == 0.0

    def test_no_solution_tags(self):
        gt = "a,b,c,d"
        answer = "I think the groups are..."
        assert score_connections(gt, answer) == 0.0


class TestUnscrambling:
    def test_perfect_order(self):
        gt = "First sentence. Second sentence. Third sentence."
        answer = "<PLOT_SUMMARY>First sentence. Second sentence. Third sentence.</PLOT_SUMMARY>"
        assert score_unscrambling(gt, answer) == 1.0

    def test_reversed_order(self):
        gt = "A. B. C."
        answer = "<PLOT_SUMMARY>C. B. A.</PLOT_SUMMARY>"
        assert score_unscrambling(gt, answer) < 1.0

    def test_empty_answer(self):
        gt = "A. B."
        assert score_unscrambling(gt, "") == 0.0
