from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class PromptSpecError(ValueError):
    """Raised when the audited Markdown prompt spec is not executable."""


@dataclass(frozen=True, slots=True)
class SearchPromptSet:
    preamble: str
    stage_a: str
    research: str
    stage_b: str
    default_charters: dict[str, str]


@dataclass(frozen=True, slots=True)
class ResearchCharter:
    research_id: str
    time_budget_minutes: float
    text: str


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    enabled_ids: tuple[str, ...]
    charters: dict[str, ResearchCharter]
    used_fallback: bool = False
    fallback_reason: str | None = None


_VALID_RESEARCH_IDS = ("R1", "R2", "R3", "R4")
_PLACEHOLDER = re.compile(r"\{[a-z][a-z0-9_]*\}")


def _code_fence_after(markdown: str, heading: str) -> str:
    marker = f"{heading}\n"
    start = markdown.find(marker)
    if start < 0:
        raise PromptSpecError(f"missing prompt heading: {heading}")
    match = re.search(r"```text\n(.*?)\n```", markdown[start + len(marker) :], re.S)
    if match is None:
        raise PromptSpecError(f"missing text code fence after: {heading}")
    return match.group(1).strip() + "\n"


def load_search_prompts(path: Path) -> SearchPromptSet:
    markdown = path.read_text(encoding="utf-8")
    prompts = SearchPromptSet(
        preamble=_code_fence_after(
            markdown, "## 二、所有 Agent 共用的 IOAI 背景前言"
        ),
        stage_a=_code_fence_after(markdown, "## 三、题目分析 Agent：阶段 A Prompt"),
        research=_code_fence_after(markdown, "## 四、Research Agent 通用 Prompt"),
        stage_b=_code_fence_after(markdown, "## 六、题目分析 Agent：阶段 B Prompt"),
        default_charters={
            "R1": _code_fence_after(markdown, "### R1：论文、现代方法与官方实现"),
            "R2": _code_fence_after(markdown, "### R2：相似任务与竞赛实践"),
            "R3": _code_fence_after(markdown, "### R3：Metric、验证、泄漏与后处理"),
            "R4": _code_fence_after(markdown, "### R4：经典路线、领域机理与反证"),
        },
    )
    required = {
        "preamble": ("{ioai_context_preamble}", False),
        "stage_a": ("{analysis_stage_a_dir}", True),
        "research": ("{research_output_path}", True),
        "stage_b": ("{final_bundle_dir}", True),
    }
    for name, (needle, must_exist) in required.items():
        value = getattr(prompts, name)
        if must_exist and needle not in value:
            raise PromptSpecError(f"{name} prompt is missing {needle}")
        if not must_exist and needle in value:
            raise PromptSpecError("shared preamble must not contain itself")
    return prompts


def render_prompt(template: str, values: dict[str, object]) -> str:
    """Render only named runtime tokens, without interpreting other braces."""

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    unresolved = sorted(set(_PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise PromptSpecError(f"unresolved prompt placeholders: {unresolved}")
    return rendered


def parse_research_plan(markdown: str) -> ResearchPlan:
    plan_match = re.search(
        r"(?mi)^\s*(?:[-*]\s*)?enabled_research_ids\s*:\s*\[([^\]]*)\]\s*$",
        markdown,
    )
    if plan_match is None:
        raise PromptSpecError("RESEARCH_CHARTERS.md lacks enabled_research_ids")

    raw_ids = plan_match.group(1).strip()
    enabled_ids = tuple(re.findall(r"\bR[1-4]\b", raw_ids))
    residue = re.sub(r"\bR[1-4]\b|[\s,'\"]", "", raw_ids)
    if residue:
        raise PromptSpecError(f"invalid enabled_research_ids content: {raw_ids!r}")
    if len(enabled_ids) != len(set(enabled_ids)):
        raise PromptSpecError("enabled_research_ids contains duplicates")

    block_matches = list(
        re.finditer(
            r"(?mi)^\s*(?:[-*]\s*)?research_id\s*:\s*(R[1-4])\s*$",
            markdown,
        )
    )
    heading_style = False
    if not block_matches and enabled_ids:
        # Claude 偶尔会把章程写成“## R1 — 标题”，同时把预算保留在顶部 YAML。
        # 接受这一等价表示，避免将有效的显式研究计划误判为默认降级计划。
        block_matches = list(
            re.finditer(r"(?mi)^\s{0,3}#{2,6}\s+(R[1-4])\b.*$", markdown)
        )
        heading_style = bool(block_matches)

    shared_budgets: dict[str, float] = {}
    if heading_style:
        prefix = markdown[: block_matches[0].start()]
        shared_budgets = {
            match.group(1): float(match.group(2))
            for match in re.finditer(
                r"(?mi)^\s+(R[1-4])\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
                prefix,
            )
        }
    charters: dict[str, ResearchCharter] = {}
    for index, match in enumerate(block_matches):
        research_id = match.group(1)
        if research_id in charters:
            raise PromptSpecError(f"duplicate charter: {research_id}")
        end = block_matches[index + 1].start() if index + 1 < len(block_matches) else len(markdown)
        block = markdown[match.start() : end].strip() + "\n"
        budget_match = re.search(
            r"(?mi)^\s*(?:[-*]\s*)?time_budget_minutes\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            block,
        )
        if budget_match is None and research_id not in shared_budgets:
            raise PromptSpecError(f"{research_id} lacks time_budget_minutes")
        budget = (
            float(budget_match.group(1))
            if budget_match is not None
            else shared_budgets[research_id]
        )
        if budget <= 0:
            raise PromptSpecError(f"{research_id} has a non-positive time budget")
        charters[research_id] = ResearchCharter(research_id, budget, block)

    if set(charters) != set(enabled_ids):
        raise PromptSpecError(
            "enabled_research_ids and charter sections differ: "
            f"enabled={enabled_ids}, charters={tuple(charters)}"
        )
    return ResearchPlan(enabled_ids=enabled_ids, charters=charters)


def fallback_research_plan(
    prompts: SearchPromptSet,
    *,
    research_window_s: float,
    reason: str,
) -> ResearchPlan:
    # 关键时间策略：并行研究共享同一个 wall-clock 窗口，不把分钟数除以 Agent 数。
    if research_window_s >= 8 * 60:
        enabled = ("R3", "R2", "R4", "R1")
    elif research_window_s >= 4 * 60:
        enabled = ("R3", "R2")
    else:
        enabled = ()
    budget_minutes = max(0.1, research_window_s / 60)
    charters = {
        research_id: ResearchCharter(
            research_id=research_id,
            time_budget_minutes=budget_minutes,
            text=(
                prompts.default_charters[research_id].rstrip()
                + f"\ntime_budget_minutes: {budget_minutes:.2f}\n"
            ),
        )
        for research_id in enabled
    }
    return ResearchPlan(
        enabled_ids=enabled,
        charters=charters,
        used_fallback=True,
        fallback_reason=reason,
    )


__all__ = [
    "PromptSpecError",
    "ResearchCharter",
    "ResearchPlan",
    "SearchPromptSet",
    "fallback_research_plan",
    "load_search_prompts",
    "parse_research_plan",
    "render_prompt",
]
