"""Shared I/O, descriptor extraction and legacy impulse-response helpers."""

import ast
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_round1/step1_audit"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def notebook_source(name):
    doc = json.loads((ROOT / "notebooks" / name).read_text())
    return "\n\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")


def sensor_groups():
    """Read only constant assignments; never execute notebook code."""
    tree = ast.parse(notebook_source("5_ablation_study_feature_models.ipynb"))
    env = {"sorted": sorted, "set": set}
    wanted = {"GAS_EXCHANGE_VENTILATION", "HR_ONLY", "HEMODYNAMICS", "NIRS",
              "ALL_SIGNALS", "NO_GAS_EXCHANGE", "HR_PLUS_NIRS", "HEMO_PLUS_NIRS", "HR_PLUS_BF"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                env[name] = eval(compile(ast.Expression(node.value), "constants", "eval"), {"__builtins__": {}}, env)
    return {
        "all_signals": env["ALL_SIGNALS"], "gas_exchange_ventilation": env["GAS_EXCHANGE_VENTILATION"],
        "hr_only": env["HR_ONLY"], "hemodynamics": env["HEMODYNAMICS"], "nirs": env["NIRS"],
        "no_gas_exchange": env["NO_GAS_EXCHANGE"], "hr_plus_nirs": env["HR_PLUS_NIRS"],
        "hemo_plus_nirs": env["HEMO_PLUS_NIRS"], "hr_plus_bf": env["HR_PLUS_BF"],
    }


def numeric_columns(headers, rows):
    result = {}
    for col in headers:
        values = []
        try:
            for row in rows:
                value = row.get(col, "")
                values.append(float(value) if value and value.lower() not in {"nan", "na", "n/a"} else float("nan"))
        except ValueError:
            continue
        result[col] = np.array(values, dtype=float)
    return result


def read_xlsx(path):
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS):
                strings.append("".join(item.itertext()))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        records = []
        for row in root.findall("m:sheetData/m:row", NS):
            record = {}
            for cell in row:
                address = re.sub(r"\d", "", cell.get("r"))
                value = cell.find("m:v", NS)
                if cell.get("t") == "inlineStr":
                    record[address] = "".join(cell.find("m:is", NS).itertext())
                elif value is not None:
                    record[address] = strings[int(value.text)] if cell.get("t") == "s" else value.text
            records.append(record)
    headers = records[0]
    rows = [{name: row.get(address, "") for address, name in headers.items()} for row in records[1:]]
    return list(headers.values()), rows


def features(columns, signals):
    time, wr = columns["Time"], columns["WorkRate"]
    values = {}
    for name in signals:
        if name not in columns:
            continue
        means = []
        for level in (30, 80):
            selected = columns[name][(time >= 300) & (time <= 1000) & (wr == level)]
            selected = selected[np.isfinite(selected)]
            mean = float(selected.mean()) if len(selected) >= 20 else float("nan")
            std = float(selected.std(ddof=1)) if len(selected) >= 20 else float("nan")
            values[f"{name}_mean_wr{level}"] = mean
            values[f"{name}_std_wr{level}"] = std
            means.append(mean)
        values[f"{name}_mean_delta_wr80_minus_wr30"] = means[1] - means[0]
    return values


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    residual = predicted - actual
    return {"N": len(actual), "MAE": float(np.abs(residual).mean()),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "R2": float(1 - np.sum(residual ** 2) / np.sum((actual - actual.mean()) ** 2)),
            "MeanResidual": float(residual.mean())}


def original_impulse(columns):
    """Reproduce the existing notebook using NumPy correlation/convolution.

    No protocol/window correction is introduced in this audit.
    """
    time, wr, vo2 = (columns[c] for c in ("Time", "WorkRate", "VO2pulm"))
    valid = np.isfinite(time) & np.isfinite(wr) & np.isfinite(vo2)
    time, wr, vo2 = time[valid], wr[valid], vo2[valid]
    order = np.argsort(time)
    time, wr, vo2 = time[order], wr[order], vo2[order]
    time = time - time.min()
    # Audit fails explicitly if this dataset would need notebook resampling.
    assert abs(np.median(np.diff(time)) - 1) <= 1e-2
    selected = (time >= 300) & (time <= 1000)
    wr, vo2 = wr[selected], vo2[selected]
    correlation = np.correlate(vo2 - vo2.mean(), wr - wr.mean(), mode="full")
    phi = {lag: correlation[len(wr) - 1 + lag] for lag in range(-150, 151)}
    h = np.zeros(150)
    for lag in range(150):
        if lag == 0:
            h[lag] = 2 * (phi[0] - phi[-20]) / (1 + 1 / 15)
        elif lag < 20:
            h[lag] = (phi[lag] - phi[-lag]) / ((0.5 + lag / 20 * (1 - lag / 40)) * (1 + 1 / 15))
        else:
            h[lag] = (phi[lag] - phi[-lag]) / (1 + 1 / 15)
    h /= h.sum()
    conv = np.convolve(wr, h)[:len(wr)]
    gain, offset = np.linalg.lstsq(np.column_stack([conv, np.ones(len(conv))]), vo2, rcond=None)[0]
    ramp_pred = gain * np.convolve(np.linspace(80, 300, 301), h)[:301] + offset
    return float(ramp_pred.max())

