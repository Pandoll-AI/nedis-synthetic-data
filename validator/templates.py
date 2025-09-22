"""Clean HTML templates for database comparison report"""

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEDIS Synthetic Data — Database Comparison Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.4;
            color: #333;
            background-color: #f8f9fa;
            padding: 15px;
            font-size: 14px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .summary {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        
        .summary h2 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .summary-card h3 {
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        
        .summary-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .content {
            padding: 30px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #495057;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .table-overview {
            overflow-x: auto;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        
        .comparison-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        
        .status-match {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-partial {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-missing {
            color: #dc3545;
            font-weight: bold;
        }
        
        .table-section {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .table-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .table-header h3 {
            color: #495057;
            margin: 0;
        }
        
        .table-content {
            padding: 20px;
        }
        
        .stat-label {
            color: #6c757d;
        }
        
        .stat-value {
            font-weight: 500;
        }
        
        .progress-bar {
            background: #e9ecef;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        
        .no-data {
            color: #6c757d;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        
        .distribution-comparison {
            margin-top: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
        }
        
        .horizontal-bar-group {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            min-height: 40px;
        }
        
        .horizontal-bars {
            display: flex;
            gap: 3px;
            align-items: center;
            flex: 1;
            max-width: 400px;
        }
        
        .bar-pair {
            display: flex;
            gap: 2px;
            align-items: center;
        }
        
        .horizontal-bar {
            height: 20px;
            min-width: 3px;
            border-radius: 3px;
            transition: width 0.3s ease;
            position: relative;
        }
        
        .bar-labels {
            margin-left: 15px;
            font-size: 11px;
            font-weight: 500;
            color: #6c757d;
        }
        
        .db1-label {
            color: #007bff;
        }
        
        .db2-label {
            color: #28a745;
        }
        
        .chart-legend {
            text-align: right;
            margin-bottom: 10px;
            font-size: 12px;
        }
        
        .legend-item {
            display: inline-block;
            margin-left: 15px;
        }
        
        .legend-color {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 5px;
            vertical-align: middle;
        }
        
        .db1-color {
            background: linear-gradient(180deg, #007bff, #0056b3);
        }
        
        .db2-color {
            background: linear-gradient(180deg, #28a745, #1e7e34);
        }
        
        .distribution-comparison h5 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 1em;
        }
        
        .distribution-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        
        .distribution-table th,
        .distribution-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        
        .distribution-table th {
            background-color: #e9ecef;
            font-weight: 600;
            color: #495057;
        }
        
        .bar-container {
            position: relative;
            display: flex;
            align-items: center;
            min-height: 20px;
        }
        
        .bar {
            height: 18px;
            border-radius: 3px;
            margin-right: 8px;
            min-width: 2px;
        }
        
        .db1-bar {
            background: linear-gradient(90deg, #007bff, #0056b3);
        }
        
        .db2-bar {
            background: linear-gradient(90deg, #28a745, #1e7e34);
        }
        
        .count-label {
            font-size: 12px;
            color: #6c757d;
            white-space: nowrap;
        }
        
        .difference {
            font-size: 12px;
            font-weight: bold;
            margin-left: 5px;
        }
        
        .diff-positive {
            color: #dc3545;
        }
        
        .diff-negative {
            color: #28a745;
        }
        
        .diff-neutral {
            color: #6c757d;
        }
        
        /* Vertical grouped bar chart styles */
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .chart-header h5 {
            margin: 0;
            color: #495057;
            font-size: 1em;
        }
        
        .chart-legend {
            display: flex;
            gap: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        
        .db1-color {
            background: linear-gradient(90deg, #007bff, #0056b3);
        }
        
        .db2-color {
            background: linear-gradient(90deg, #28a745, #1e7e34);
        }
        
        .vertical-chart {
            margin: 15px 0;
            background: white;
            border-radius: 4px;
            padding: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        
        .vertical-bar-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 80px;
        }
        
        .value-label {
            font-size: 13px;
            font-weight: 500;
            min-width: 80px;
            margin-right: 15px;
            text-align: right;
        }
        
        .vertical-bars {
            display: flex;
            gap: 2px;
            align-items: end;
            height: 150px;
        }
        
        .bar-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 27px;
        }
        
        .vertical-bar {
            width: 25px;
            min-height: 4px;
            border-radius: 3px 3px 0 0;
            transition: height 0.3s ease;
            margin-bottom: 5px;
        }
        
        .db1-bar {
            background: linear-gradient(180deg, #007bff, #0056b3);
        }
        
        .db2-bar {
            background: linear-gradient(180deg, #28a745, #1e7e34);
        }
        
        .bar-label {
            font-size: 11px;
            font-weight: 500;
            color: #6c757d;
            text-align: center;
            line-height: 1.2;
            max-width: 60px;
        }
        
        .diff-indicator {
            margin-top: 8px;
            text-align: center;
        }
        
        .diff-label {
            font-weight: bold;
            font-size: 11px;
        }
        
        .distribution-table-container {
            margin-top: 15px;
        }
        
        .column-comparison {
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .column-header {
            background: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #e9ecef;
            font-size: 15px;
        }
        
        .column-stats {
            padding: 15px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-group {
            background: white;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        
        .stat-group h4 {
            color: #495057;
            margin-bottom: 8px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 13px;
        }
        
        @media (max-width: 768px) {
            .column-stats {
                grid-template-columns: 1fr;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NEDIS Synthetic Data — Database Comparison</h1>
            <p>Original vs Synthetic: {{db1_name}} ↔ {{db2_name}}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Common tables</h3>
                    <div class="value">{{common_tables_count}}</div>
                </div>
                <div class="summary-card">
                    <h3>Only in {{db1_name}}</h3>
                    <div class="value">{{db1_only_count}}</div>
                </div>
                <div class="summary-card">
                    <h3>Only in {{db2_name}}</h3>
                    <div class="value">{{db2_only_count}}</div>
                </div>
                <div class="summary-card">
                    <h3>Total comparisons</h3>
                    <div class="value">{{total_comparisons}}</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>Table overview</h2>
                <div class="table-overview">
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Table Name</th>
                                <th>{{db1_name}}</th>
                                <th>{{db2_name}}</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {{table_overview_rows}}
                        </tbody>
                    </table>
                </div>
            </div>
            
            {{table_sections}}
        </div>
    </div>
    
</body>
</html>
'''

def format_number(num):
    """Format numbers for display"""
    if num is None:
        return 'N/A'
    if isinstance(num, float):
        if num == int(num):
            return str(int(num))
        else:
            return f"{num:.2f}"
    return str(num)

def format_percentage(num):
    """Format percentage for display"""
    if num is None:
        return 'N/A'
    return f"{num:.1f}%"

def create_distribution_comparison(db1_stats, db2_stats, db1_name="Database 1", db2_name="Database 2"):
    """Create distribution comparison visualization"""
    if not db1_stats and not db2_stats:
        return ''
    
    # For categorical data with top values, create comparison table
    if ((db1_stats and db1_stats.top_values) or (db2_stats and db2_stats.top_values)):
        return create_categorical_distribution(db1_stats, db2_stats, db1_name, db2_name)
    
    # For numerical data, create range comparison
    if ((db1_stats and db1_stats.mean_val is not None) or 
        (db2_stats and db2_stats.mean_val is not None)):
        return create_numerical_comparison(db1_stats, db2_stats, db1_name, db2_name)
    
    return ''

def create_categorical_distribution(db1_stats, db2_stats, db1_name="Database 1", db2_name="Database 2"):
    """Create categorical value distribution comparison with stacked percentage bars"""
    # Collect all unique values from both databases
    all_values = set()
    db1_values = {}
    db2_values = {}
    
    if db1_stats and db1_stats.top_values:
        for value, count in db1_stats.top_values:
            all_values.add(value)
            db1_values[value] = count
    
    if db2_stats and db2_stats.top_values:
        for value, count in db2_stats.top_values:
            all_values.add(value)
            db2_values[value] = count
    
    if not all_values:
        return ''
    
    # Calculate percentages for each database
    db1_total = db1_stats.total_count if db1_stats else 0
    db2_total = db2_stats.total_count if db2_stats else 0
    
    # Sort values by combined frequency
    sorted_values = sorted(all_values, key=lambda x: -(db1_values.get(x, 0) + db2_values.get(x, 0)))
    
    # Create grouped horizontal bar chart
    chart_rows = []
    table_rows = []
    
    # Find max percentage for scaling
    max_pct = 0
    for value in sorted_values:
        db1_count = db1_values.get(value, 0)
        db2_count = db2_values.get(value, 0)
        db1_pct = (db1_count / db1_total * 100) if db1_total > 0 else 0
        db2_pct = (db2_count / db2_total * 100) if db2_total > 0 else 0
        max_pct = max(max_pct, db1_pct, db2_pct)
    
    for value in sorted_values:
        db1_count = db1_values.get(value, 0)
        db2_count = db2_values.get(value, 0)
        
        # Calculate percentages
        db1_pct = (db1_count / db1_total * 100) if db1_total > 0 else 0
        db2_pct = (db2_count / db2_total * 100) if db2_total > 0 else 0
        
        # Calculate bar widths relative to max percentage
        db1_bar_width = (db1_pct / max_pct * 100) if max_pct > 0 else 0
        db2_bar_width = (db2_pct / max_pct * 100) if max_pct > 0 else 0
        
        # Percentage difference
        diff_pct = db2_pct - db1_pct
        diff_class = 'diff-positive' if diff_pct > 0 else 'diff-negative' if diff_pct < 0 else 'diff-neutral'
        
        chart_rows.append(f'''
        <div class="horizontal-bar-group">
            <div class="value-label"><strong>{value}</strong></div>
            <div class="horizontal-bars">
                <div class="bar-pair">
                    <div class="horizontal-bar db1-bar" style="width: {db1_bar_width}%" title="{db1_name}: {db1_pct:.1f}%"></div>
                    <div class="horizontal-bar db2-bar" style="width: {db2_bar_width}%" title="{db2_name}: {db2_pct:.1f}%"></div>
                </div>
            </div>
            <div class="bar-labels">
                <span class="db1-label">{db1_pct:.1f}% ({db1_count:,})</span> | 
                <span class="db2-label">{db2_pct:.1f}% ({db2_count:,})</span>
                <span class="diff-label {diff_class}"> Δ{diff_pct:+.1f}%</span>
            </div>
        </div>
        ''')
        
        # Also create table row for detailed numbers
        table_rows.append(f'''
        <tr>
            <td><strong>{value}</strong></td>
            <td>{db1_count:,} ({db1_pct:.1f}%)</td>
            <td>{db2_count:,} ({db2_pct:.1f}%)</td>
            <td><span class="difference {diff_class}">{diff_pct:+.1f}%</span></td>
        </tr>
        ''')
    
    return f'''
    <div class="distribution-comparison">
        <div class="chart-header">
            <h5>Value distribution comparison</h5>
            <div class="chart-legend">
                <div class="legend-item">
                    <div class="legend-color db1-color"></div>
                    <span>{db1_name}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color db2-color"></div>
                    <span>{db2_name}</span>
                </div>
            </div>
        </div>
        
        <!-- Horizontal grouped bar chart -->
        <div class="horizontal-chart">
            {''.join(chart_rows)}
        </div>
        
        <!-- Detailed table -->
        <div class="distribution-table-container">
            <table class="distribution-table">
                <thead>
                    <tr>
                        <th>Value</th>
                        <th>{db1_name}</th>
                        <th>{db2_name}</th>
                        <th>Δ Difference</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
    </div>
    '''

def create_numerical_comparison(db1_stats, db2_stats, db1_name="Database 1", db2_name="Database 2"):
    """Create numerical statistics comparison"""
    stats_comparison = []
    
    stats_to_compare = [
        ('Mean', 'mean_val'),
        ('Std Dev', 'std_val'),
        ('Min', 'min_val'),
        ('Max', 'max_val')
    ]
    
    for label, attr in stats_to_compare:
        db1_val = getattr(db1_stats, attr) if db1_stats else None
        db2_val = getattr(db2_stats, attr) if db2_stats else None
        
        if db1_val is not None or db2_val is not None:
            # Calculate percentage difference if both values exist
            diff_text = ''
            if db1_val is not None and db2_val is not None and db1_val != 0:
                diff_pct = ((db2_val - db1_val) / db1_val) * 100
                diff_class = 'diff-positive' if diff_pct > 0 else 'diff-negative' if diff_pct < 0 else 'diff-neutral'
                diff_text = f'<span class="difference {diff_class}">({diff_pct:+.1f}%)</span>'
            
            stats_comparison.append(f'''
            <tr>
                <td><strong>{label}</strong></td>
                <td>{format_number(db1_val)}</td>
                <td>{format_number(db2_val)} {diff_text}</td>
            </tr>
            ''')
    
    if not stats_comparison:
        return ''
    
    return f'''
    <div class="distribution-comparison">
        <h5>Statistical Comparison</h5>
        <table class="distribution-table">
            <thead>
                <tr>
                    <th>Statistic</th>
                    <th>{db1_name}</th>
                    <th>{db2_name}</th>
                </tr>
            </thead>
            <tbody>
                {''.join(stats_comparison)}
            </tbody>
        </table>
    </div>
    '''

def get_status_class(status):
    """Get CSS class for status"""
    if status == 'match':
        return 'status-match'
    elif status == 'partial':
        return 'status-partial'
    else:
        return 'status-missing'

def create_table_overview_row(table_name, db1_info, db2_info, status):
    """Create a table overview row"""
    db1_cell = f"{db1_info['rows']:,} rows, {db1_info['cols']} cols" if db1_info else '-'
    db2_cell = f"{db2_info['rows']:,} rows, {db2_info['cols']} cols" if db2_info else '-'
    
    status_text = {
        'match': '✓ Match',
        'partial': '⚠ Partial',
        'missing': '✗ Missing'
    }.get(status, '· Unknown')
    
    return f'''
    <tr>
        <td><strong>{table_name}</strong></td>
        <td>{db1_cell}</td>
        <td>{db2_cell}</td>
        <td><span class="{get_status_class(status)}">{status_text}</span></td>
    </tr>
    '''

def create_column_comparison(column_name, db1_stats, db2_stats, db1_name, db2_name):
    """Create column comparison section with distribution comparison"""
    
    # Create distribution comparison
    distribution_section = create_distribution_comparison(db1_stats, db2_stats, db1_name, db2_name)
    
    db1_content = ''
    db2_content = ''
    
    if db1_stats:
        db1_content = f'''
        <div class="stat-group">
            <h4>{db1_name} • {db1_stats.data_type}</h4>
            <div class="stat-item">
                <span class="stat-label">Total rows:</span>
                <span class="stat-value">{db1_stats.total_count:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Non-null rows:</span>
                <span class="stat-value">{db1_stats.total_count - db1_stats.null_count:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Null rows (%):</span>
                <span class="stat-value">{db1_stats.null_count:,} ({format_percentage(db1_stats.null_percentage)})</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Unique values:</span>
                <span class="stat-value">{db1_stats.unique_count:,}</span>
            </div>
            {'<div class="stat-item"><span class="stat-label">Mean:</span><span class="stat-value">' + format_number(db1_stats.mean_val) + '</span></div>' if db1_stats.mean_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Std Dev:</span><span class="stat-value">' + format_number(db1_stats.std_val) + '</span></div>' if db1_stats.std_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Min:</span><span class="stat-value">' + format_number(db1_stats.min_val) + '</span></div>' if db1_stats.min_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Max:</span><span class="stat-value">' + format_number(db1_stats.max_val) + '</span></div>' if db1_stats.max_val is not None else ''}
        </div>
        '''
    else:
        db1_content = f'<div class="stat-group"><h4>{db1_name}</h4><div class="no-data">Column not found</div></div>'
    
    if db2_stats:
        db2_content = f'''
        <div class="stat-group">
            <h4>{db2_name} • {db2_stats.data_type}</h4>
            <div class="stat-item">
                <span class="stat-label">Total rows:</span>
                <span class="stat-value">{db2_stats.total_count:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Non-null rows:</span>
                <span class="stat-value">{db2_stats.total_count - db2_stats.null_count:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Null rows (%):</span>
                <span class="stat-value">{db2_stats.null_count:,} ({format_percentage(db2_stats.null_percentage)})</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Unique values:</span>
                <span class="stat-value">{db2_stats.unique_count:,}</span>
            </div>
            {'<div class="stat-item"><span class="stat-label">Mean:</span><span class="stat-value">' + format_number(db2_stats.mean_val) + '</span></div>' if db2_stats.mean_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Std Dev:</span><span class="stat-value">' + format_number(db2_stats.std_val) + '</span></div>' if db2_stats.std_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Min:</span><span class="stat-value">' + format_number(db2_stats.min_val) + '</span></div>' if db2_stats.min_val is not None else ''}
            {'<div class="stat-item"><span class="stat-label">Max:</span><span class="stat-value">' + format_number(db2_stats.max_val) + '</span></div>' if db2_stats.max_val is not None else ''}
        </div>
        '''
    else:
        db2_content = f'<div class="stat-group"><h4>{db2_name}</h4><div class="no-data">Column not found</div></div>'
    
    return f'''
    <div class="column-comparison">
        <div class="column-header">
            <strong>{column_name}</strong>
        </div>
        <div class="column-stats">
            {db1_content}
            {db2_content}
        </div>
        {distribution_section}
    </div>
    '''

def create_table_section(table_name, db1_info, db1_stats, db2_info, db2_stats, db1_name, db2_name):
    """Create a complete table section"""
    # Get all column names from both databases
    db1_columns = set(db1_stats.keys()) if db1_stats else set()
    db2_columns = set(db2_stats.keys()) if db2_stats else set()
    all_columns = sorted(db1_columns.union(db2_columns))
    
    column_comparisons = []
    for column in all_columns:
        db1_col_stats = db1_stats.get(column) if db1_stats else None
        db2_col_stats = db2_stats.get(column) if db2_stats else None
        column_comparisons.append(create_column_comparison(column, db1_col_stats, db2_col_stats, db1_name, db2_name))
    
    return f'''
    <div class="table-section">
        <div class="table-header">
            <h3>{table_name}</h3>
        </div>
        <div class="table-content">
            {''.join(column_comparisons) if column_comparisons else '<div class="no-data">No column data available</div>'}
        </div>
    </div>
    '''
