import json
from jinja2 import Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AIDRIN Concept Report</title>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        h2 { border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2980b9; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .llm-box { background-color: #e8f4f8; padding: 20px; border-left: 5px solid #3498db; margin-top: 20px;}
    </style>
</head>
<body>
    <div class="container">
        <h1>AIDRIN Readiness Report</h1>
        <p><strong>Source:</strong> {{ data.source }}</p>
        <p><strong>Rows:</strong> {{ data.row_count }} | <strong>Columns:</strong> {{ data.column_count }}</p>
        
        <h2>1. Traditional Data Quality</h2>
        <pre>{{ traditional_metrics_json }}</pre>
        
        <h2>2. AI-Specific Readiness</h2>
        <pre>{{ ai_readiness_metrics_json }}</pre>
        
        <h2>3. Data Privacy & Compliance</h2>
        <pre>{{ privacy_metrics_json }}</pre>
        
        <h2>4. FAIR Metadata Compliance</h2>
        <pre>{{ fair_compliance_json }}</pre>
        
        <h2>5. Actionable Intelligence (LLM)</h2>
        <div class="llm-box">
            {{ data.llm_insights | replace('\n', '<br>') | safe }}
        </div>
    </div>
</body>
</html>
"""

class HTMLReporter:
    """Generates an HTML report from AIDRIN profile results."""
    
    @staticmethod
    def generate(report_data: dict, output_path: str):
        template = Template(HTML_TEMPLATE)
        
        html_content = template.render(
            data=report_data,
            traditional_metrics_json=json.dumps(report_data.get('traditional_metrics', {}), indent=2),
            ai_readiness_metrics_json=json.dumps(report_data.get('ai_readiness_metrics', {}), indent=2),
            privacy_metrics_json=json.dumps(report_data.get('privacy_metrics', {}), indent=2),
            fair_compliance_json=json.dumps(report_data.get('fair_compliance', {}), indent=2)
        )
        
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"HTML Report generated at {output_path}")
