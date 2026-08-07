#!/usr/bin/env python3
"""依据冻结的 TASK_CONTRACT.json 检查 IOAI Kernel package，不修改候选目录。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-ant-api[0-9A-Za-z_-]{8,}", re.IGNORECASE),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?:KAGGLE_(?:API_)?(?:TOKEN|KEY)|OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY|"
        r"(?:API|SECRET|ACCESS)[_-]?(?:KEY|TOKEN))\s*[:=]\s*['\"]?[^'\"\s]{12,}",
        re.IGNORECASE,
    ),
)
CUDA1_RE = re.compile(r"cuda\s*:\s*[1-9]", re.IGNORECASE)
SET_DEVICE_RE = re.compile(r"cuda\.set_device\s*\(([^)]+)\)", re.IGNORECASE)
VISIBLE_ASSIGN_RE = re.compile(r"CUDA_VISIBLE_DEVICES(?:['\"]?\])?\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
MULTI_GPU_RE = re.compile(r"(?:DataParallel|DistributedDataParallel|device_ids\s*=|cuda\.device_count\s*\()")
CUDA_METHOD_INDEX_RE = re.compile(r"\.cuda\s*\(\s*[1-9]", re.IGNORECASE)
TORCH_DEVICE_INDEX_RE = re.compile(r"torch\.device\s*\(\s*['\"]cuda['\"]\s*,\s*[1-9]", re.IGNORECASE)
REPORT_END_RE = re.compile(r"(?:environment\s+setup|环境设置|环境初始化)", re.IGNORECASE)
REPORT_NUMBER_RE = re.compile(r"technical\s+report\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
UNKNOWN_VALUES = {None, "", "UNKNOWN", "unknown", "TBD", "tbd"}
PLACEHOLDER_RE = re.compile(r"(?:UNKNOWN|TBD|TODO|REPLACE(?:_ME|_WITH)?|<[^>]+>)", re.IGNORECASE)
FINAL_TRAINING_SPLITS_RE = re.compile(
    r"^#\s*IOAI_FINAL_TRAINING_SPLITS:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SUSPICIOUS_ARTIFACT_NAME_RE = re.compile(
    r"(?:label|answer|pred(?:iction)?|lookup|embedding|checkpoint|weight|transcript|"
    r"feature|cache|target|ground[_-]?truth|gold|y[_-]?val|val[_-]?y|test[_-]?y)",
    re.IGNORECASE,
)
LITERAL_UNPACK_CALLS = {"b64decode", "loads", "decompress", "unpack", "frombuffer"}


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return None


def report_paragraph_count(source: str) -> tuple[int, int]:
    blocks = 0
    boundary_line = 0
    in_block = False
    numbered: list[tuple[int, int]] = []
    for idx, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            in_block = False
            continue
        if stripped.startswith("#!") or stripped.startswith("# -*-"):
            continue
        if not stripped.startswith("#"):
            boundary_line = idx
            break
        content = stripped[1:].strip()
        if REPORT_END_RE.search(content):
            boundary_line = idx
            break
        marker = REPORT_NUMBER_RE.search(content)
        if marker:
            numbered.append((int(marker.group(1)), int(marker.group(2))))
        if not content or set(content) <= {"=", "-", "_"}:
            in_block = False
            continue
        if not in_block:
            blocks += 1
            in_block = True
    if numbered:
        totals = {total for _, total in numbered}
        if len(totals) == 1:
            total = next(iter(totals))
            if {index for index, _ in numbered} == set(range(1, total + 1)):
                return total, boundary_line
        return -1, boundary_line
    return blocks, boundary_line


def require_value(contract: dict[str, Any], key: str) -> Any:
    value = contract.get(key)
    if value is None or (isinstance(value, str) and value in UNKNOWN_VALUES):
        raise RuntimeError(f"合同关键字段 {key} 未冻结")
    return value


def string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or PLACEHOLDER_RE.search(item) for item in value
    ):
        raise RuntimeError(f"合同字段 {key} 必须是字符串列表，可为空")
    return value


def parse_timestamp(value: Any, key: str) -> str:
    if not isinstance(value, str) or PLACEHOLDER_RE.search(value):
        raise RuntimeError(f"合同字段 {key} 必须是带时区 ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"合同字段 {key} 不是 ISO-8601") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"合同字段 {key} 必须带时区")
    return value


def file_contains_secret(path: Path) -> bool:
    tail = ""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1_048_576):
                text = tail + chunk.decode("utf-8", errors="ignore")
                if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                    return True
                tail = text[-512:]
    except OSError:
        return False
    return False


def violates_only_cuda0(source: str) -> bool:
    if (
        CUDA1_RE.search(source)
        or MULTI_GPU_RE.search(source)
        or CUDA_METHOD_INDEX_RE.search(source)
        or TORCH_DEVICE_INDEX_RE.search(source)
    ):
        return True
    if any(match.group(1).strip() != "0" for match in VISIBLE_ASSIGN_RE.finditer(source)):
        return True
    return any(match.group(1).strip() != "0" for match in SET_DEVICE_RE.finditer(source))


def training_split_declaration(source: str) -> list[str] | None:
    matches = FINAL_TRAINING_SPLITS_RE.findall(source)
    if len(matches) != 1:
        return None
    values = [value.strip() for value in matches[0].split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        return None
    return values


def _literal_collection_size(node: ast.AST) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_collection_size(node.left)
        right = _literal_collection_size(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            base = _literal_collection_size(node.left)
            if base is not None:
                return base * max(0, node.right.value)
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, int):
            base = _literal_collection_size(node.right)
            if base is not None:
                return base * max(0, node.left.value)
    return None


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _largest_literal_collection(node: ast.AST) -> int | None:
    sizes = [
        size
        for child in ast.walk(node)
        for size in [_literal_collection_size(child)]
        if size is not None
    ]
    return max(sizes, default=None)


def embedded_artifact_findings(source: str) -> list[str]:
    """拒绝明显内嵌答案/权重；这是审计门，不是任意 Python 的语义证明。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 8192:
            findings.add("发现超过 8192 字节的内嵌字符串/bytes")
        size = _literal_collection_size(node)
        if size is not None and size >= 256:
            findings.add("发现至少 256 项的内嵌 literal collection")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assigned_size = _largest_literal_collection(value) if value is not None else None
            if (
                assigned_size is not None
                and assigned_size >= 32
                and any(SUSPICIOUS_ARTIFACT_NAME_RE.search(name) for name in _assigned_names(node))
            ):
                findings.add("疑似标签/预测/权重变量含至少 32 项硬编码 literal")
        if isinstance(node, ast.Call):
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if (
                function_name in LITERAL_UNPACK_CALLS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, (str, bytes))
            ):
                findings.add("发现从内嵌 literal 解码/反序列化数据")
    return sorted(findings)


def load_contract(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取合同 {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("TASK_CONTRACT.json 顶层必须是对象")

    mode = require_value(raw, "mode")
    if mode not in {"official", "rehearsal"}:
        raise RuntimeError("合同 mode 必须是 official 或 rehearsal")
    critical = (
        "competition_slug",
        "official_sources",
        "official_labeled_splits",
        "asset_audit",
        "final_training_splits",
        "start_time_utc",
        "submission_deadline_utc",
        "metric",
        "submission_schema",
        "task_validator",
        "notebook_version_limit",
        "kernel_timeout_seconds",
        "latest_start_safety_seconds",
        "submit_command_safety_seconds",
        "late_report_window_seconds",
        "allowed_hardware",
        "kernel_slots",
        "allowed_sources",
        "allowed_source_provenance",
        "agent_research_network_policy",
        "kernel_internet_enabled",
        "required_report",
        "strict_kernel_files",
        "final_selection_policy",
    )
    for key in critical:
        require_value(raw, key)

    competition = raw["competition_slug"]
    if not isinstance(competition, str) or PLACEHOLDER_RE.search(competition):
        raise RuntimeError("competition_slug 仍含占位符")
    raw["start_time_utc"] = parse_timestamp(raw["start_time_utc"], "start_time_utc")
    raw["submission_deadline_utc"] = parse_timestamp(raw["submission_deadline_utc"], "submission_deadline_utc")

    metric = raw["metric"]
    if not isinstance(metric, dict) or not isinstance(metric.get("name"), str) or PLACEHOLDER_RE.search(metric["name"]):
        raise RuntimeError("metric 必须含非占位 name")
    if metric.get("direction") not in {"maximize", "minimize"}:
        raise RuntimeError("metric.direction 必须是 maximize 或 minimize")

    schema = raw["submission_schema"]
    if not isinstance(schema, dict):
        raise RuntimeError("submission_schema 必须是对象")
    sample = schema.get("sample_submission")
    if not isinstance(sample, str) or PLACEHOLDER_RE.search(sample):
        raise RuntimeError("submission_schema.sample_submission 必须是本地文件路径")
    sample_path = Path(sample)
    if not sample_path.is_absolute():
        sample_path = path.parent / sample_path
    sample_path = sample_path.resolve()
    if mode == "official" and not sample_path.is_relative_to(path.parent.resolve()):
        raise RuntimeError("official sample submission 必须位于 IOAI_TASK_ROOT 内")
    if not sample_path.is_file():
        raise RuntimeError(f"sample submission 不存在：{sample_path}")
    sample_sha = schema.get("sample_submission_sha256")
    if not isinstance(sample_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sample_sha):
        raise RuntimeError("submission_schema.sample_submission_sha256 必须是 64 位 SHA")
    if hashlib.sha256(sample_path.read_bytes()).hexdigest().lower() != sample_sha.lower():
        raise RuntimeError("sample submission SHA 与合同不一致")
    try:
        with sample_path.open(newline="", encoding="utf-8-sig") as handle:
            sample_header = next(csv.reader(handle))
    except (OSError, StopIteration, csv.Error) as exc:
        raise RuntimeError("无法读取 sample submission header") from exc
    if not sample_header or len(sample_header) != len(set(sample_header)):
        raise RuntimeError("sample submission header 为空或含重复列")
    output_filename = schema.get("output_filename")
    if (
        not isinstance(output_filename, str)
        or not output_filename
        or PLACEHOLDER_RE.search(output_filename)
        or Path(output_filename).name != output_filename
    ):
        raise RuntimeError("submission_schema.output_filename 必须是单个文件名")
    id_columns = string_list(schema.get("id_columns"), "submission_schema.id_columns")
    if not id_columns:
        raise RuntimeError("submission_schema.id_columns 不得为空")
    numeric_columns = string_list(schema.get("numeric_columns"), "submission_schema.numeric_columns")
    integer_columns = string_list(schema.get("integer_columns"), "submission_schema.integer_columns")
    for key in ("allow_row_reorder", "allow_column_reorder"):
        if as_bool(schema.get(key)) is None:
            raise RuntimeError(f"submission_schema.{key} 必须是布尔值")
    ranges = schema.get("ranges")
    if not isinstance(ranges, dict):
        raise RuntimeError("submission_schema.ranges 必须是对象")
    normalized_ranges: dict[str, list[float]] = {}
    for column, bounds in ranges.items():
        if (
            not isinstance(column, str)
            or PLACEHOLDER_RE.search(column)
            or not isinstance(bounds, list)
            or len(bounds) != 2
        ):
            raise RuntimeError("submission_schema.ranges 必须是 COLUMN: [MIN, MAX]")
        try:
            normalized_ranges[column] = [float(bounds[0]), float(bounds[1])]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("submission_schema.ranges 边界必须是数值") from exc
        if normalized_ranges[column][0] > normalized_ranges[column][1]:
            raise RuntimeError(f"submission_schema.ranges {column} 下界大于上界")
    declared_columns = set(id_columns) | set(numeric_columns) | set(integer_columns) | set(normalized_ranges)
    missing_columns = sorted(declared_columns - set(sample_header))
    if missing_columns:
        raise RuntimeError(f"submission_schema 声明了 sample 中不存在的列：{missing_columns}")
    raw["submission_schema"] = {
        **schema,
        "sample_submission": str(sample_path.resolve()),
        "id_columns": id_columns,
        "numeric_columns": numeric_columns,
        "integer_columns": integer_columns,
        "allow_row_reorder": bool(as_bool(schema["allow_row_reorder"])),
        "allow_column_reorder": bool(as_bool(schema["allow_column_reorder"])),
        "ranges": normalized_ranges,
    }

    task_validator = raw["task_validator"]
    if not isinstance(task_validator, dict):
        raise RuntimeError("task_validator 必须是对象")
    validator_path_value = task_validator.get("path")
    validator_sha = task_validator.get("sha256")
    if not isinstance(validator_path_value, str) or PLACEHOLDER_RE.search(validator_path_value):
        raise RuntimeError("task_validator.path 必须是本地 Python 文件路径")
    validator_path = Path(validator_path_value)
    if not validator_path.is_absolute():
        validator_path = path.parent / validator_path
    validator_path = validator_path.resolve()
    if mode == "official" and not validator_path.is_relative_to(path.parent.resolve()):
        raise RuntimeError("official task_validator 必须位于 IOAI_TASK_ROOT 内")
    if not validator_path.is_file() or validator_path.suffix != ".py":
        raise RuntimeError(f"task_validator 必须是存在的 .py 文件：{validator_path}")
    if not isinstance(validator_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", validator_sha):
        raise RuntimeError("task_validator.sha256 必须是 64 位 SHA")
    if hashlib.sha256(validator_path.read_bytes()).hexdigest().lower() != validator_sha.lower():
        raise RuntimeError("task_validator SHA 与合同不一致")
    try:
        compile(validator_path.read_text(encoding="utf-8"), str(validator_path), "exec")
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError(f"task_validator 无法作为 Python 编译：{exc}") from exc
    raw["task_validator"] = {**task_validator, "path": str(validator_path.resolve())}

    sources = raw["official_sources"]
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("official_sources 必须至少含一份官方原文")
    for source in sources:
        if not isinstance(source, dict) or not source.get("url") or not source.get("retrieved_at") or not source.get("local_path"):
            raise RuntimeError("每个 official_sources 项必须含 url、retrieved_at 与 local_path")
        if PLACEHOLDER_RE.search(str(source["url"])):
            raise RuntimeError("official_sources.url 仍含占位符")
        parse_timestamp(source["retrieved_at"], "official_sources.retrieved_at")
        sha = source.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            raise RuntimeError("每个 official_sources 项必须含 64 位 sha256")
        if PLACEHOLDER_RE.search(str(source["local_path"])):
            raise RuntimeError("official_sources.local_path 仍含占位符")
        local_path = Path(str(source["local_path"]))
        if not local_path.is_absolute():
            local_path = path.parent / local_path
        local_path = local_path.resolve()
        if mode == "official" and not local_path.is_relative_to(path.parent.resolve()):
            raise RuntimeError("official source 快照必须位于 IOAI_TASK_ROOT 内")
        if not local_path.is_file():
            raise RuntimeError(f"official source 本地快照不存在：{local_path}")
        actual_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
        if actual_sha.lower() != sha.lower():
            raise RuntimeError(f"official source SHA 不匹配：{local_path}")
        source["local_path"] = str(local_path.resolve())

    source_hashes = {str(source["sha256"]).lower() for source in sources}
    labeled_splits = raw["official_labeled_splits"]
    if not isinstance(labeled_splits, list):
        raise RuntimeError("official_labeled_splits 必须是列表；没有时填 []")
    normalized_splits: list[dict[str, Any]] = []
    split_names: set[str] = set()
    for split in labeled_splits:
        if not isinstance(split, dict):
            raise RuntimeError("official_labeled_splits 每项必须是对象")
        name = split.get("name")
        labels_path_value = split.get("labels_path")
        labels_sha = split.get("labels_sha256")
        label_columns = string_list(split.get("label_columns"), "official_labeled_splits.label_columns")
        purpose = split.get("official_purpose")
        training_use = split.get("training_use")
        provenance_sha = split.get("source_sha256")
        if (
            not isinstance(name, str)
            or not name
            or PLACEHOLDER_RE.search(name)
            or name in split_names
        ):
            raise RuntimeError("official_labeled_splits.name 必须非占位且唯一")
        split_names.add(name)
        if not isinstance(labels_path_value, str) or PLACEHOLDER_RE.search(labels_path_value):
            raise RuntimeError(f"官方带标签 split {name} 的 labels_path 非法")
        labels_path = Path(labels_path_value)
        if not labels_path.is_absolute():
            labels_path = path.parent / labels_path
        labels_path = labels_path.resolve()
        if mode == "official" and not labels_path.is_relative_to(path.parent.resolve()):
            raise RuntimeError(f"官方带标签 split {name} 必须位于 IOAI_TASK_ROOT 内")
        if not labels_path.is_file():
            raise RuntimeError(f"官方带标签 split {name} 的标签文件不存在：{labels_path}")
        if not isinstance(labels_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", labels_sha):
            raise RuntimeError(f"官方带标签 split {name} 必须含 64 位 labels_sha256")
        if hashlib.sha256(labels_path.read_bytes()).hexdigest().lower() != labels_sha.lower():
            raise RuntimeError(f"官方带标签 split {name} 的标签 SHA 不匹配")
        if not label_columns:
            raise RuntimeError(f"官方带标签 split {name} 的 label_columns 不得为空")
        if not isinstance(purpose, str) or not purpose or PLACEHOLDER_RE.search(purpose):
            raise RuntimeError(f"官方带标签 split {name} 必须记录 official_purpose")
        if training_use not in {"allowed", "validation_only", "UNKNOWN"}:
            raise RuntimeError(
                f"官方带标签 split {name} 的 training_use 必须是 allowed、validation_only 或 UNKNOWN"
            )
        if not isinstance(provenance_sha, str) or provenance_sha.lower() not in source_hashes:
            raise RuntimeError(f"官方带标签 split {name} 的 source_sha256 必须引用 official_sources")
        try:
            sample_count = int(split.get("sample_count"))
            group_count = int(split.get("group_count"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"官方带标签 split {name} 必须记录 sample_count/group_count") from exc
        if sample_count <= 0 or group_count < 0:
            raise RuntimeError(f"官方带标签 split {name} 的 sample_count 必须 >0，group_count 必须 >=0")
        normalized_splits.append({
            **split,
            "labels_path": str(labels_path),
            "labels_sha256": labels_sha.lower(),
            "label_columns": label_columns,
            "sample_count": sample_count,
            "group_count": group_count,
            "source_sha256": provenance_sha.lower(),
        })
    raw["official_labeled_splits"] = normalized_splits

    asset_audit = raw["asset_audit"]
    if not isinstance(asset_audit, dict):
        raise RuntimeError("asset_audit 必须是对象")
    manifest_value = asset_audit.get("manifest_path")
    manifest_sha = asset_audit.get("manifest_sha256")
    if not isinstance(manifest_value, str) or PLACEHOLDER_RE.search(manifest_value):
        raise RuntimeError("asset_audit.manifest_path 必须是本地 JSON 路径")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    if mode == "official" and not manifest_path.is_relative_to(path.parent.resolve()):
        raise RuntimeError("official asset manifest 必须位于 IOAI_TASK_ROOT 内")
    if not manifest_path.is_file() or manifest_path.suffix.lower() != ".json":
        raise RuntimeError(f"asset manifest 不存在或不是 JSON：{manifest_path}")
    if not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_sha):
        raise RuntimeError("asset_audit.manifest_sha256 必须是 64 位 SHA")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest().lower() != manifest_sha.lower():
        raise RuntimeError("asset manifest SHA 与合同不一致")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("asset manifest 无法解析") from exc
    if not isinstance(manifest, dict) or as_bool(manifest.get("discovery_complete")) is not True:
        raise RuntimeError("asset manifest 必须明确 discovery_complete=true")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("asset manifest.files 必须是非空列表")
    file_splits: set[str] = set()
    file_paths: set[str] = set()
    manifest_file_records: list[dict[str, Any]] = []
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or PLACEHOLDER_RE.search(str(item.get("path")))
            or str(item.get("path")) in file_paths
            or not isinstance(item.get("role"), str)
            or not item.get("role")
            or not isinstance(item.get("split"), str)
            or not item.get("split")
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(item.get("sha256")))
        ):
            raise RuntimeError("asset manifest.files 每项必须含唯一 path、role、split、64位 sha256")
        file_paths.add(str(item["path"]))
        file_splits.add(str(item["split"]))
        manifest_file_records.append(item)
    base_training_splits = string_list(
        manifest.get("base_training_split_names"),
        "asset_manifest.base_training_split_names",
    )
    if not base_training_splits:
        raise RuntimeError("asset manifest 至少要声明一个 base training split")
    manifest_labeled = string_list(
        manifest.get("labeled_split_names"),
        "asset_manifest.labeled_split_names",
    )
    if set(manifest_labeled) != split_names or len(manifest_labeled) != len(split_names):
        raise RuntimeError("asset manifest 的 labeled_split_names 与 official_labeled_splits 不一致")
    for split in normalized_splits:
        if not any(
            str(item["split"]) == str(split["name"])
            and str(item["sha256"]).lower() == str(split["labels_sha256"]).lower()
            and "label" in str(item["role"]).lower()
            for item in manifest_file_records
        ):
            raise RuntimeError(f"asset manifest 未以相同 SHA 登记带标签 split {split['name']} 的标签文件")
    required_file_splits = set(base_training_splits) | split_names | {"submission_test"}
    if not required_file_splits.issubset(file_splits):
        raise RuntimeError("asset manifest.files 未覆盖 base train、带标签 split 或 submission_test")
    overlap_checks = manifest.get("overlap_checks")
    if not isinstance(overlap_checks, list):
        raise RuntimeError("asset manifest.overlap_checks 必须是列表")
    checked_pairs: set[frozenset[str]] = set()
    for check in overlap_checks:
        if not isinstance(check, dict):
            raise RuntimeError("asset overlap check 必须是对象")
        left = check.get("left")
        right = check.get("right")
        authorized = as_bool(check.get("label_transfer_authorized"))
        if (
            not isinstance(left, str)
            or not left
            or not isinstance(right, str)
            or not right
            or left == right
            or authorized is None
        ):
            raise RuntimeError("asset overlap check 必须声明不同的 left/right 与布尔授权")
        try:
            id_overlap = int(check.get("id_overlap_count"))
            content_overlap = int(check.get("content_overlap_count"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("asset overlap check 必须记录非负 overlap count") from exc
        if id_overlap < 0 or content_overlap < 0:
            raise RuntimeError("asset overlap count 不得为负")
        pair = frozenset((left, right))
        if pair in checked_pairs:
            raise RuntimeError("asset overlap check 不得重复")
        checked_pairs.add(pair)
        if "submission_test" in pair and (id_overlap > 0 or content_overlap > 0):
            authorization_sha = check.get("source_sha256")
            if not authorized or not isinstance(authorization_sha, str) or authorization_sha.lower() not in source_hashes:
                raise RuntimeError("带标签 split 与 submission_test 重叠时必须有官方原文明确授权")
    required_pairs = {
        frozenset((labeled, other))
        for labeled in split_names
        for other in [*base_training_splits, "submission_test"]
        if labeled != other
    }
    if not required_pairs.issubset(checked_pairs):
        raise RuntimeError("asset manifest 缺少带标签 split 对 train/submission_test 的 ID+内容重叠检查")

    final_training_splits = string_list(raw["final_training_splits"], "final_training_splits")
    if not final_training_splits or len(final_training_splits) != len(set(final_training_splits)):
        raise RuntimeError("final_training_splits 必须非空且不重复")
    known_training_splits = set(base_training_splits) | split_names
    unknown_training_splits = sorted(set(final_training_splits) - known_training_splits)
    if unknown_training_splits:
        raise RuntimeError(f"final_training_splits 含资产清单未声明的 split：{unknown_training_splits}")
    split_training_use = {str(item["name"]): item["training_use"] for item in normalized_splits}
    forbidden_training_splits = sorted(
        name
        for name in final_training_splits
        if name in split_training_use and split_training_use[name] != "allowed"
    )
    if forbidden_training_splits:
        raise RuntimeError(
            "validation_only/UNKNOWN split 不得进入 final_training_splits："
            + ", ".join(forbidden_training_splits)
        )
    raw["asset_audit"] = {
        **asset_audit,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha.lower(),
    }
    raw["final_training_splits"] = final_training_splits

    try:
        timeout = int(raw["kernel_timeout_seconds"])
        version_limit = int(raw["notebook_version_limit"])
        safety = int(raw["latest_start_safety_seconds"])
        submit_safety = int(raw["submit_command_safety_seconds"])
        late_report_window = int(raw["late_report_window_seconds"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("timeout 和 version limit 必须是正整数") from exc
    if (
        timeout <= 0
        or version_limit <= 0
        or safety < (180 if mode == "official" else 0)
        or submit_safety < (30 if mode == "official" else 0)
        or late_report_window < 0
        or (
            late_report_window > 0
            and late_report_window <= timeout + safety + submit_safety
        )
    ):
        raise RuntimeError(
            "timeout/version limit 必须为正；official push/submit safety 至少 180/30 秒；"
            "非零 late Report 窗口必须容纳 timeout 与两段 safety"
        )

    hardware = raw["allowed_hardware"]
    if not isinstance(hardware, dict):
        raise RuntimeError("allowed_hardware 必须是对象")
    allow_cpu = as_bool(hardware.get("allow_cpu"))
    only_cuda0 = as_bool(hardware.get("only_cuda0"))
    gpu_shapes = string_list(hardware.get("gpu_shapes"), "allowed_hardware.gpu_shapes")
    if allow_cpu is None or only_cuda0 is None or (not allow_cpu and not gpu_shapes):
        raise RuntimeError("allowed_hardware 必须明确 allow_cpu、gpu_shapes、only_cuda0，并至少允许一种硬件")

    slots = raw["kernel_slots"]
    if not isinstance(slots, dict):
        raise RuntimeError("kernel_slots 必须是对象")
    slot_keys = (
        "max_concurrent_cpu",
        "max_concurrent_gpu",
        "per_actor_inflight",
        "final_priority_window_seconds",
        "final_waiter_ttl_seconds",
        "orphan_reservation_seconds",
    )
    try:
        normalized_slots = {key: int(slots[key]) for key in slot_keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("kernel_slots 缺少字段或包含非整数") from exc
    if (
        normalized_slots["max_concurrent_cpu"] < 0
        or normalized_slots["max_concurrent_gpu"] < 0
        or normalized_slots["per_actor_inflight"] <= 0
        or normalized_slots["final_priority_window_seconds"] < 0
        or normalized_slots["final_waiter_ttl_seconds"] < 15
        or normalized_slots["orphan_reservation_seconds"] < (300 if mode == "official" else 60)
    ):
        raise RuntimeError(
            "kernel_slots 容量不得为负，per_actor 必须为正，waiter TTL 至少15秒；"
            "official orphan TTL 至少300秒（rehearsal 至少60秒）"
        )
    if allow_cpu and normalized_slots["max_concurrent_cpu"] <= 0:
        raise RuntimeError("当前允许 CPU，但 kernel_slots.max_concurrent_cpu 未提供正容量")
    if gpu_shapes and normalized_slots["max_concurrent_gpu"] <= 0:
        raise RuntimeError("当前允许 GPU，但 kernel_slots.max_concurrent_gpu 未提供正容量")
    if normalized_slots["per_actor_inflight"] > max(
        normalized_slots["max_concurrent_cpu"],
        normalized_slots["max_concurrent_gpu"],
    ):
        raise RuntimeError("kernel_slots.per_actor_inflight 大于所有资源容量")

    allowed_sources = raw["allowed_sources"]
    if not isinstance(allowed_sources, dict):
        raise RuntimeError("allowed_sources 必须是对象")
    normalized_sources = {
        field: string_list(allowed_sources.get(field), f"allowed_sources.{field}")
        for field in ("competition_sources", "dataset_sources", "kernel_sources", "model_sources")
    }
    if raw["competition_slug"] not in normalized_sources["competition_sources"]:
        raise RuntimeError("allowed_sources.competition_sources 必须包含 competition_slug")
    provenance = raw["allowed_source_provenance"]
    if not isinstance(provenance, list):
        raise RuntimeError("allowed_source_provenance 必须是列表")
    expected_provenance = {
        (kind, ref)
        for kind, refs in normalized_sources.items()
        for ref in refs
        if not (kind == "competition_sources" and ref == raw["competition_slug"])
    }
    seen_provenance: set[tuple[str, str]] = set()
    for item in provenance:
        if not isinstance(item, dict):
            raise RuntimeError("allowed_source_provenance 每项必须是对象")
        kind = item.get("kind")
        ref = item.get("ref")
        authorization = item.get("authorization")
        authorization_sha = item.get("source_sha256")
        key = (str(kind), str(ref))
        if (
            key not in expected_provenance
            or key in seen_provenance
            or authorization not in {"organizer_provided", "rules_explicitly_allowed"}
            or not isinstance(authorization_sha, str)
            or authorization_sha.lower() not in source_hashes
        ):
            raise RuntimeError(
                "每个额外 competition/dataset/kernel/model source 必须唯一绑定官方原文和授权类型"
            )
        seen_provenance.add(key)
    if seen_provenance != expected_provenance:
        missing = sorted(expected_provenance - seen_provenance)
        raise RuntimeError(f"allowed_source_provenance 缺少官方授权：{missing}")

    if raw["agent_research_network_policy"] not in {"allowed_methods_only", "disabled"}:
        raise RuntimeError("agent_research_network_policy 必须是 allowed_methods_only 或 disabled")
    if raw["final_selection_policy"] not in {"kaggle_auto", "manual"}:
        raise RuntimeError("final_selection_policy 必须是 kaggle_auto 或 manual")

    for key in ("kernel_internet_enabled", "required_report", "strict_kernel_files"):
        if as_bool(raw[key]) is None:
            raise RuntimeError(f"合同字段 {key} 必须是布尔值")
    if mode == "official" and as_bool(raw["strict_kernel_files"]) is not True:
        raise RuntimeError("official 模式必须启用 strict_kernel_files，禁止随 Kernel 打包本地产物")

    return {
        **raw,
        "kernel_timeout_seconds": timeout,
        "notebook_version_limit": version_limit,
        "latest_start_safety_seconds": safety,
        "submit_command_safety_seconds": submit_safety,
        "late_report_window_seconds": late_report_window,
        "allowed_hardware": {
            "allow_cpu": bool(allow_cpu),
            "gpu_shapes": gpu_shapes,
            "only_cuda0": bool(only_cuda0),
        },
        "kernel_slots": normalized_slots,
        "allowed_sources": normalized_sources,
        "allowed_source_provenance": provenance,
        "kernel_internet_enabled": bool(as_bool(raw["kernel_internet_enabled"])),
        "required_report": bool(as_bool(raw["required_report"])),
        "strict_kernel_files": bool(as_bool(raw["strict_kernel_files"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_dir", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.kernel_dir.resolve()
    try:
        contract = load_contract(args.contract.resolve())
    except RuntimeError as exc:
        print(f"PRECHECK FAILED\n- ERROR: {exc}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        print(f"ERROR: Kernel 目录不存在：{root}", file=sys.stderr)
        return 2

    metadata_path = root / "kernel-metadata.json"
    if not metadata_path.is_file():
        errors.append("缺少 kernel-metadata.json")
        metadata: dict[str, Any] = {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"kernel-metadata.json 无法解析：{exc}")
            metadata = {}

    code_name = metadata.get("code_file")
    code_path = (root / str(code_name)).resolve() if code_name else None
    if not code_name:
        errors.append("metadata 缺少 code_file")
    elif code_path and not code_path.is_relative_to(root):
        errors.append(f"metadata code_file 逃逸 Kernel 目录：{code_name}")
        code_path = None
    elif not code_path or not code_path.is_file():
        errors.append(f"metadata code_file 不存在：{code_name}")
    elif code_path.suffix != ".py":
        errors.append("metadata code_file 必须是 .py 文件")

    if str(metadata.get("language", "")).lower() != "python":
        errors.append("metadata language 必须是 python")
    if str(metadata.get("kernel_type", "")).lower() != "script":
        errors.append("metadata kernel_type 必须是 script")
    if as_bool(metadata.get("is_private")) is not True:
        errors.append("metadata is_private 必须为 true")
    expected_internet = contract["kernel_internet_enabled"]
    if as_bool(metadata.get("enable_internet")) is not expected_internet:
        errors.append(f"metadata enable_internet 必须与合同一致：{expected_internet}")

    gpu = as_bool(metadata.get("enable_gpu"))
    shape = str(metadata.get("machine_shape") or "")
    hardware = contract["allowed_hardware"]
    if gpu is True:
        if shape not in hardware["gpu_shapes"]:
            errors.append(f"GPU machine_shape {shape!r} 不在合同列表 {hardware['gpu_shapes']!r}")
    elif gpu is False:
        if not hardware["allow_cpu"]:
            errors.append("当前合同不允许 CPU Kernel")
        if shape:
            errors.append("CPU Kernel 的 machine_shape 应为空")
    else:
        errors.append("metadata enable_gpu 必须明确为 true 或 false")

    for field, allowed in contract["allowed_sources"].items():
        values = metadata.get(field, [])
        if not isinstance(values, list):
            errors.append(f"metadata {field} 必须是列表")
            continue
        unexpected = sorted(str(value) for value in values if str(value) not in set(allowed))
        missing = sorted(value for value in allowed if value not in values) if field == "competition_sources" else []
        if unexpected:
            errors.append(f"metadata {field} 含合同未允许来源：{unexpected}")
        if missing:
            errors.append(f"metadata {field} 缺少合同要求来源：{missing}")

    files = [path for path in root.rglob("*") if path.is_file()]
    relative_files = [path.relative_to(root) for path in files]
    symlinks = [str(path) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        errors.append("Kernel package 不允许符号链接：" + ", ".join(symlinks))
    forbidden = [str(path) for path in relative_files if "__pycache__" in path.parts or path.suffix == ".pyc"]
    if forbidden:
        errors.append("存在缓存文件：" + ", ".join(forbidden))
    if contract["strict_kernel_files"]:
        expected = {Path("kernel-metadata.json")}
        if code_name:
            expected.add(Path(str(code_name)))
        extras = sorted(str(path) for path in set(relative_files) - expected)
        missing = sorted(str(path) for path in expected - set(relative_files))
        if extras:
            errors.append("严格双文件合同发现额外文件：" + ", ".join(extras))
        if missing:
            errors.append("严格双文件合同缺少文件：" + ", ".join(missing))

    for path in files:
        if file_contains_secret(path):
            errors.append(f"{path.relative_to(root)} 含疑似硬编码凭证（已隐藏具体值）")

    python_files = [path for path in files if path.suffix == ".py"]
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(root)} Python 语法失败：{exc.msg}（第 {exc.lineno} 行）")
        if path == code_path and contract["mode"] == "official":
            declared_training_splits = training_split_declaration(source)
            if declared_training_splits != contract["final_training_splits"]:
                errors.append(
                    f"{path.relative_to(root)} 必须唯一声明 "
                    "# IOAI_FINAL_TRAINING_SPLITS: ...，并与合同完全一致"
                )
            for finding in embedded_artifact_findings(source):
                errors.append(f"{path.relative_to(root)} 禁止内嵌本地标签/预测/权重：{finding}")
        if gpu is True and hardware["only_cuda0"] and violates_only_cuda0(source):
            errors.append(f"{path.relative_to(root)} 疑似使用非零或多 GPU；当前合同只允许 cuda:0")
        paragraphs, boundary_line = report_paragraph_count(source)
        if contract["required_report"] and not 8 <= paragraphs <= 10:
            errors.append(f"{path.relative_to(root)} 顶部 Report 应有 8–10 段，当前检测到 {paragraphs} 段")
        if contract["required_report"] and boundary_line <= 1:
            errors.append(f"{path.relative_to(root)} Report 必须位于代码和 environment setup 之前")
        if not contract["required_report"] and path == code_path and not 8 <= paragraphs <= 10:
            warnings.append(f"顶部 Report 检测到 {paragraphs} 段；当前合同未要求 Report")

    if errors:
        print("PRECHECK FAILED")
        for error in errors:
            print(f"- ERROR: {error}")
        for warning in warnings:
            print(f"- WARNING: {warning}")
        return 1

    print("PRECHECK PASSED")
    for warning in warnings:
        print(f"- WARNING: {warning}")
    print(f"- kernel_dir: {root}")
    print(f"- competition: {contract['competition_slug']}")
    print(f"- timeout: {contract['kernel_timeout_seconds']}")
    print(f"- command: kaggle kernels push -p {root} --timeout {contract['kernel_timeout_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
