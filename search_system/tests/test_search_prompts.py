from __future__ import annotations

import unittest
from pathlib import Path

from ioai_agent_system.search_prompts import (
    PromptSpecError,
    load_search_prompts,
    parse_research_plan,
    render_prompt,
)


class SearchPromptTests(unittest.TestCase):
    def test_audited_prompt_document_is_executable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prompts = load_search_prompts(root / "SEARCH_PROMPTS_ZH.md")
        self.assertIn("AI Models Track 背景", prompts.preamble)
        self.assertIn("{analysis_stage_a_dir}", prompts.stage_a)
        self.assertEqual(set(prompts.default_charters), {"R1", "R2", "R3", "R4"})

    def test_research_plan_requires_exact_enabled_ids_and_charters(self) -> None:
        markdown = """research_plan:
  enabled_research_ids: [R3, R2]
  reason: test

research_id: R3
time_budget_minutes: 5
primary_scope: metric

research_id: R2
time_budget_minutes: 4.5
primary_scope: similar tasks
"""
        plan = parse_research_plan(markdown)
        self.assertEqual(plan.enabled_ids, ("R3", "R2"))
        self.assertEqual(plan.charters["R2"].time_budget_minutes, 4.5)

        with self.assertRaises(PromptSpecError):
            parse_research_plan(markdown.replace("[R3, R2]", "[R3]"))

    def test_zero_research_plan_and_unresolved_placeholder(self) -> None:
        plan = parse_research_plan(
            "research_plan:\n  enabled_research_ids: []\n  reason: enough local evidence\n"
        )
        self.assertEqual(plan.enabled_ids, ())
        with self.assertRaises(PromptSpecError):
            render_prompt("hello {missing_value}", {})


if __name__ == "__main__":
    unittest.main()
