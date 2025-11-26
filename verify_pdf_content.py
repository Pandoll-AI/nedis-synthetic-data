#!/usr/bin/env python3
"""
Quick script to verify the generated PDF content
"""

import sys
import os
import glob

def verify_pdf_content():
    """Verify the generated test PDF"""
    print("🔍 PDF Content Verification")
    print("=" * 40)

    # Find the test PDF
    pdf_files = glob.glob("test_pdf_export_*.pdf")

    if not pdf_files:
        print("❌ No test PDF found")
        return False

    pdf_file = pdf_files[0]  # Get the latest one
    print(f"📄 Analyzing: {pdf_file}")

    # Basic file info
    file_size = os.path.getsize(pdf_file)
    print(f"📏 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    try:
        import PyPDF2
        print("📚 Reading PDF with PyPDF2...")

        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            print(f"📖 Pages: {len(reader.pages)}")
            print(f"📋 Metadata: {reader.metadata if reader.metadata else 'None'}")

            # Try to extract some text from first page
            if len(reader.pages) > 0:
                first_page = reader.pages[0]
                text = first_page.extract_text()

                print("📝 First page content preview:")
                print("-" * 30)
                lines = text.split('\n')[:10]  # First 10 lines
                for line in lines:
                    if line.strip():
                        print(f"   {line.strip()}")
                print("-" * 30)

                # Check for expected content
                expected_terms = [
                    "NEDIS",
                    "Comparison Report",
                    "nedis2017",
                    "Original Database",
                    "Synthetic Database"
                ]

                found_terms = []
                for term in expected_terms:
                    if term.lower() in text.lower():
                        found_terms.append(term)

                print(f"✅ Expected content found: {', '.join(found_terms)}")
                print(f"📊 Content verification: {len(found_terms)}/{len(expected_terms)} terms found")

                if len(found_terms) >= 3:
                    print("🎉 PDF content verification PASSED!")
                    return True
                else:
                    print("⚠️ PDF content verification incomplete")
                    return False

    except ImportError:
        print("ℹ️ PyPDF2 not available - basic verification only")
        print("💡 Install with: pip install PyPDF2")

        # Just check if it's a valid PDF file
        with open(pdf_file, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'%PDF-'):
                print("✅ File appears to be a valid PDF")
                return True
            else:
                print("❌ File does not appear to be a valid PDF")
                return False

    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

if __name__ == "__main__":
    success = verify_pdf_content()
    sys.exit(0 if success else 1)