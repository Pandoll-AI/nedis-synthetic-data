# Database Comparison Validator

Simple tool to compare two DuckDB databases and generate a clean HTML report.

## Features

- **Table Overview**: Side-by-side comparison of database structures
- **Column Analysis**: Detailed statistics for each column (null%, unique count, mean, min/max)  
- **Clean Design**: Single scrolling page with embedded CSS
- **No Dependencies**: Uses only Python stdlib, DuckDB, and pandas

## Usage

```bash
python report.py database1.duckdb database2.duckdb --output report.html
```

### Options

- `--output` or `-o`: Specify output HTML file (default: comparison_report.html)
- `--verbose` or `-v`: Enable verbose output for debugging

### Example

```bash
python report.py ../nedis_data.duckdb ../nedis_synth_2017.duckdb --output comparison.html
```

## Files

- `report.py`: Main script with CLI interface
- `db_analyzer.py`: Database analysis utilities
- `templates.py`: HTML template and formatting functions

## Requirements

- Python 3.7+
- DuckDB Python package
- pandas (for basic statistics)