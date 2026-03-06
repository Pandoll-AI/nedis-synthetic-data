#!/usr/bin/env python3
"""
patterns.json을 zlib 압축 + Base64 인코딩하여 HTML 템플릿에 삽입합니다.
난독화 효과(소스에서 패턴 식별 불가) + 파일 크기 절감.
"""

import base64
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "nedis_generator_template.html"
PATTERNS_PATH = PROJECT_ROOT / "outputs" / "html_generator" / "patterns.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "html_generator" / "nedis_generator.html"

PLACEHOLDER = "%%PATTERNS_BASE64%%"


def main():
    print(f"[1/4] Reading template: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    print(f"[2/4] Reading patterns: {PATTERNS_PATH}")
    patterns_json = PATTERNS_PATH.read_text(encoding="utf-8")
    raw_kb = len(patterns_json.encode("utf-8")) / 1024
    print(f"       Raw size: {raw_kb:.1f} KB")

    print(f"[3/4] Compressing (zlib + base64)...")
    compressed = zlib.compress(patterns_json.encode("utf-8"), level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    encoded_kb = len(encoded) / 1024
    ratio = (1 - encoded_kb / raw_kb) * 100
    print(f"       Compressed: {encoded_kb:.1f} KB ({ratio:.0f}% reduction)")

    if PLACEHOLDER not in template:
        raise ValueError(f"Placeholder '{PLACEHOLDER}' not found in template")

    print(f"[4/4] Building final HTML: {OUTPUT_PATH}")
    final_html = template.replace(PLACEHOLDER, encoded)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(final_html, encoding="utf-8")

    total_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n[DONE] Output: {OUTPUT_PATH}")
    print(f"       Total size: {total_kb:.1f} KB")
    print(f"       Patterns (compressed): {encoded_kb:.1f} KB (raw: {raw_kb:.1f} KB)")
    print(f"       HTML/JS: {total_kb - encoded_kb:.1f} KB")


if __name__ == "__main__":
    main()
