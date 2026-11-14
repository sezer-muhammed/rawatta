#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""YAML checks, as run by .github/workflows/ci.yml (and locally, from any directory).

1. Every *.yml / *.yaml file under .github/, and CITATION.cff, parses as YAML.
2. The issue forms in .github/ISSUE_TEMPLATE/ follow the structure GitHub renders
   (https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms):
   name / description / body, known element types, unique ids, labels on input elements,
   options on dropdowns, string values where GitHub expects strings. GitHub does not render a
   form that breaks these rules; actionlint only checks the workflows.
3. .github/ISSUE_TEMPLATE/config.yml has the documented keys.

Needs PyYAML.  usage: scripts/ci/check_yaml.py   (or: python3 scripts/ci/check_yaml.py)
"""

import argparse
import re
import sys
from pathlib import Path


def parse_args(argv=None):
    """--help prints the description above; the check itself takes no arguments."""
    parser = argparse.ArgumentParser(
        prog="scripts/ci/check_yaml.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parse_args()  # before the PyYAML import, so --help works without it

try:
    import yaml
except ImportError:
    sys.exit("check_yaml.py needs PyYAML (pip install pyyaml)")

ROOT = Path(__file__).resolve().parents[2]
ELEMENT_TYPES = {"markdown", "textarea", "input", "dropdown", "checkboxes"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STRING_KEYS = ("label", "description", "placeholder", "value", "render")


def yaml_files():
    files = sorted(p for p in (ROOT / ".github").rglob("*") if p.suffix in (".yml", ".yaml"))
    cff = ROOT / "CITATION.cff"
    if cff.is_file():
        files.append(cff)
    return files


def check_form(doc, where, errors):
    def err(msg):
        errors.append(f"{where}: {msg}")

    if not isinstance(doc, dict):
        return err("an issue form must be a mapping")
    for key in ("name", "description"):
        if not isinstance(doc.get(key), str) or not doc[key].strip():
            err(f"top-level '{key}' must be a non-empty string")
    body = doc.get("body")
    if not isinstance(body, list) or not body:
        return err("'body' must be a non-empty list")
    ids, inputs = set(), 0
    for i, el in enumerate(body):
        at = f"body[{i}]"
        if not isinstance(el, dict):
            err(f"{at} must be a mapping")
            continue
        kind = el.get("type")
        if kind not in ELEMENT_TYPES:
            err(f"{at}: type {kind!r} is not one of {sorted(ELEMENT_TYPES)}")
            continue
        attrs = el.get("attributes")
        if not isinstance(attrs, dict):
            err(f"{at}: 'attributes' must be a mapping")
            continue
        for key in STRING_KEYS:
            if key in attrs and not isinstance(attrs[key], str):
                err(f"{at}: attributes.{key} must be a string, got {type(attrs[key]).__name__}")
        if "id" in el:
            if not isinstance(el["id"], str) or not ID_RE.match(el["id"]):
                err(f"{at}: id {el['id']!r} may only use letters, digits, '-' and '_'")
            elif el["id"] in ids:
                err(f"{at}: duplicate id {el['id']!r}")
            else:
                ids.add(el["id"])
        if kind == "markdown":
            if not isinstance(attrs.get("value"), str) or not attrs["value"].strip():
                err(f"{at}: markdown needs a non-empty attributes.value")
            continue
        inputs += 1
        if not isinstance(attrs.get("label"), str) or not attrs["label"].strip():
            err(f"{at}: {kind} needs a non-empty attributes.label")
        if kind == "dropdown":
            opts = attrs.get("options")
            if not isinstance(opts, list) or not opts or not all(isinstance(o, str) for o in opts):
                err(f"{at}: dropdown needs a non-empty list of string options")
            elif len(set(opts)) != len(opts):
                err(f"{at}: dropdown options must be unique")
        if kind == "checkboxes":
            opts = attrs.get("options")
            if not isinstance(opts, list) or not opts or not all(
                isinstance(o, dict) and isinstance(o.get("label"), str) for o in opts
            ):
                err(f"{at}: checkboxes need a list of options with a string label")
        val = el.get("validations", {})
        if not isinstance(val, dict) or not isinstance(val.get("required", False), bool):
            err(f"{at}: validations.required must be true or false")
    if inputs == 0:
        err("a form needs at least one element that is not markdown")


def check_config(doc, where, errors):
    if not isinstance(doc, dict):
        errors.append(f"{where}: must be a mapping")
        return
    if "blank_issues_enabled" in doc and not isinstance(doc["blank_issues_enabled"], bool):
        errors.append(f"{where}: blank_issues_enabled must be true or false")
    for i, link in enumerate(doc.get("contact_links") or []):
        if not isinstance(link, dict) or not all(
            isinstance(link.get(k), str) and link[k].strip() for k in ("name", "url", "about")
        ):
            errors.append(f"{where}: contact_links[{i}] needs string name, url and about")


def main():
    errors = []
    files = yaml_files()
    forms = 0
    for path in files:
        where = path.relative_to(ROOT).as_posix()
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"{where}: not valid YAML: {e}".replace("\n", " "))
            continue
        if path.parent.name == "ISSUE_TEMPLATE" and path.parent.parent.name == ".github":
            if path.stem == "config":
                check_config(doc, where, errors)
            else:
                forms += 1
                check_form(doc, where, errors)
    for e in errors:
        print(f"error: {e}")
    print(f"checked {len(files)} YAML files ({forms} issue forms)")
    if errors:
        return 1
    print("ok: YAML checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
