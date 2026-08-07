#!/usr/bin/env python3
"""把远端真实 submission.csv 与 sample submission 做通用结构和数值校验。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"{path} 没有 header")
            return list(reader.fieldnames), list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"无法读取 {path}: {exc}") from exc


def parse_range(value: str) -> tuple[str, float, float]:
    try:
        column, low, high = value.split(":", 2)
        return column, float(low), float(high)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--range 格式应为 COLUMN:MIN:MAX") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="从 Kaggle Kernel output 下载的真实 submission.csv")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--id-column", action="append", required=True, help="ID 列，可重复")
    parser.add_argument("--allow-row-reorder", action="store_true")
    parser.add_argument("--allow-column-reorder", action="store_true")
    parser.add_argument("--numeric-column", action="append", default=[])
    parser.add_argument("--integer-column", action="append", default=[])
    parser.add_argument("--range", dest="ranges", action="append", type=parse_range, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    try:
        actual_header, actual = read_csv(args.actual)
        sample_header, sample = read_csv(args.sample)
    except RuntimeError as exc:
        print(f"SUBMISSION INVALID\n- ERROR: {exc}")
        return 1

    if len(actual_header) != len(set(actual_header)):
        errors.append(f"actual header 含重复列：{actual_header}")
    if len(sample_header) != len(set(sample_header)):
        errors.append(f"sample header 含重复列：{sample_header}")
    malformed_rows = [index for index, row in enumerate(actual, start=2) if None in row or any(value is None for value in row.values())]
    if malformed_rows:
        errors.append(f"actual 含多余或缺失字段，首个异常行：{malformed_rows[0]}")
    if args.allow_column_reorder:
        if set(actual_header) != set(sample_header):
            errors.append(f"列集合不一致：actual={actual_header}, sample={sample_header}")
    elif actual_header != sample_header:
        errors.append(f"header/列顺序不一致：actual={actual_header}, sample={sample_header}")
    if len(actual) != len(sample):
        errors.append(f"行数不一致：actual={len(actual)}, sample={len(sample)}")
    for column in args.id_column:
        if column not in sample_header or column not in actual_header:
            errors.append(f"ID 列不存在：{column}")

    if not errors:
        actual_ids = [tuple(row[column] for column in args.id_column) for row in actual]
        sample_ids = [tuple(row[column] for column in args.id_column) for row in sample]
        if len(set(actual_ids)) != len(actual_ids):
            errors.append("actual 含重复 ID")
        if args.allow_row_reorder:
            if set(actual_ids) != set(sample_ids):
                errors.append("actual 与 sample 的 ID 集合不一致")
        elif actual_ids != sample_ids:
            errors.append("actual 与 sample 的 ID 顺序不一致")

    numeric = set(args.numeric_column) | set(args.integer_column)
    ranges = {column: (low, high) for column, low, high in args.ranges}
    for column in numeric | set(ranges):
        if column not in actual_header:
            errors.append(f"数值约束列不存在：{column}")
            continue
        for row_number, row in enumerate(actual, start=2):
            raw = row.get(column, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"第 {row_number} 行 {column} 不是数值")
                break
            if not math.isfinite(value):
                errors.append(f"第 {row_number} 行 {column} 是 NaN/Inf")
                break
            if column in args.integer_column and not value.is_integer():
                errors.append(f"第 {row_number} 行 {column} 不是整数")
                break
            if column in ranges:
                low, high = ranges[column]
                if not low <= value <= high:
                    errors.append(f"第 {row_number} 行 {column}={value} 超出 [{low}, {high}]")
                    break

    if errors:
        print("SUBMISSION INVALID")
        for error in errors:
            print(f"- ERROR: {error}")
        return 1
    print("SUBMISSION VALID")
    print(f"- rows: {len(actual)}")
    print(f"- columns: {actual_header}")
    print(f"- unique_ids: {len(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
