#!/usr/bin/env python3
"""
patterns.json을 HTML 템플릿에 인라인으로 삽입하여 완전한 단일 HTML 파일을 생성합니다.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "nedis_generator_template.html"
PATTERNS_PATH = PROJECT_ROOT / "outputs" / "html_generator" / "patterns.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "html_generator" / "nedis_generator.html"

PLACEHOLDER = "%%PATTERNS_JSON%%"


def main():
    print(f"[1/3] Reading template: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    print(f"[2/3] Reading patterns: {PATTERNS_PATH}")
    patterns_json = PATTERNS_PATH.read_text(encoding="utf-8")
    patterns_kb = len(patterns_json) / 1024
    print(f"       Patterns size: {patterns_kb:.1f} KB")

    if PLACEHOLDER not in template:
        raise ValueError(f"Placeholder '{PLACEHOLDER}' not found in template")

    print(f"[3/3] Building final HTML: {OUTPUT_PATH}")
    final_html = template.replace(PLACEHOLDER, patterns_json)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(final_html, encoding="utf-8")

    total_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n[DONE] Output: {OUTPUT_PATH}")
    print(f"       Total size: {total_kb:.1f} KB")
    print(f"       Patterns: {patterns_kb:.1f} KB")
    print(f"       HTML/JS: {total_kb - patterns_kb:.1f} KB")


if __name__ == "__main__":
    main()
