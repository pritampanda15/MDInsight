"""
HTML Report Generator - Assembles all MDInsight analyses into a single
interactive HTML dashboard report.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    raise ImportError("plotly is required.")

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive HTML report from MDInsight analysis.

    Parameters
    ----------
    title : str
        Report title.
    output_path : str or Path
        Output HTML file path.
    """

    def __init__(self, title: str = "MDInsight Deep Dive Report", output_path: str = "report.html"):
        self.title = title
        self.output_path = Path(output_path)
        self._sections: List[Dict] = []

    def add_section(self, title: str, description: str = "", figures: Optional[List[go.Figure]] = None, text_content: str = ""):
        """Add a section to the report."""
        self._sections.append({
            "title": title,
            "description": description,
            "figures": figures or [],
            "text_content": text_content,
        })

    def add_summary_table(self, title: str, data: Dict):
        """Add a key-value summary table."""
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in data.items()
        )
        table_html = f"""
        <table style="max-width:700px;">
            {rows}
        </table>
        """
        self._sections.append({
            "title": title,
            "description": "",
            "figures": [],
            "text_content": table_html,
        })

    def add_stat_cards(self, title: str, cards: List[Dict[str, str]], description: str = "", grid_cols: int = 3):
        """Add a row of stat cards."""
        grid_class = f"grid-{grid_cols}" if grid_cols in (2, 3) else "grid-3"
        cards_html = f'<div class="{grid_class}">'
        for card in cards:
            color = card.get("color", "#3498db")
            cards_html += f"""
            <div class="stat-card" style="border-left-color: {color};">
                <div class="value">{card['value']}</div>
                <div class="label">{card['label']}</div>
            </div>"""
        cards_html += "</div>"

        self._sections.append({
            "title": title,
            "description": description,
            "figures": [],
            "text_content": cards_html,
        })

    def add_insight_box(self, text: str, box_type: str = "insight", label: str = "AI Insight"):
        """Append an insight or anomaly box to the last section."""
        css_class = "anomaly-box" if box_type == "anomaly" else "insight-box"
        html = f'<div class="{css_class}"><strong>{label}:</strong> {text}</div>'
        if self._sections:
            self._sections[-1]["text_content"] += html
        else:
            self._sections.append({"title": "", "description": "", "figures": [], "text_content": html})

    def add_html_content(self, html: str):
        """Append arbitrary HTML to the last section."""
        if self._sections:
            self._sections[-1]["text_content"] += html
        else:
            self._sections.append({"title": "", "description": "", "figures": [], "text_content": html})

    def add_data_table(self, headers: List[str], rows: List[List[str]], title: str = ""):
        """Append a formatted data table to the last section."""
        header_html = "<tr>" + "".join(
            f'<td style="font-weight:700;color:#c0392b;">{h}</td>' for h in headers
        ) + "</tr>"
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        table_html = ""
        if title:
            table_html += f'<h4 style="color:#2c3e50;margin:1em 0 0.3em;">{title}</h4>'
        table_html += f"""
        <table style="max-width:700px;">
            {header_html}
            {rows_html}
        </table>
        """
        if self._sections:
            self._sections[-1]["text_content"] += table_html
        else:
            self._sections.append({"title": "", "description": "", "figures": [], "text_content": table_html})

    def add_figures_to_last_section(self, figures: List[go.Figure]):
        """Append figures to the most recent section."""
        if self._sections and figures:
            self._sections[-1]["figures"].extend(figures)

    def generate(self) -> Path:
        """Generate the complete HTML report."""
        sections_html = []
        plotlyjs_included = False

        for i, section in enumerate(self._sections):
            figs_html = ""
            for fig in section["figures"]:
                try:
                    figs_html += f'<div class="plot-container">{pio.to_html(fig, full_html=False, include_plotlyjs=(not plotlyjs_included))}</div>'
                    plotlyjs_included = True
                except Exception as e:
                    figs_html += f"<p style='color:red;'>Figure error: {e}</p>"

            # Skip rendering sections with no title as standalone (they were appended)
            title = section['title']
            if not title:
                # Append content to previous section
                if sections_html:
                    prev = sections_html[-1]
                    insert_point = prev.rfind("</section>")
                    if insert_point >= 0:
                        extra = section.get('text_content', '') + figs_html
                        sections_html[-1] = prev[:insert_point] + extra + prev[insert_point:]
                continue

            sections_html.append(f"""
            <section id="section-{i}">
                <h2>{section['title']}</h2>
                {f"<p class='desc'>{section['description']}</p>" if section['description'] else ""}
                {section.get('text_content', '')}
                {figs_html}
            </section>
            """)

        toc_entries = "".join(
            f'<a href="#section-{i}">{section["title"]}</a>'
            for i, section in enumerate(self._sections) if section["title"]
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 2em;
            background: #fafafa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 2em 2.5em;
            border-radius: 12px;
            margin-bottom: 2em;
        }}
        .header h1 {{ margin: 0; font-size: 2em; }}
        .header p {{ margin: 0.5em 0 0; opacity: 0.9; font-size: 0.95em; }}
        section {{
            background: white;
            padding: 2em 2.5em;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 2em;
        }}
        section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5em;
            margin-bottom: 0.8em;
            font-size: 1.4em;
        }}
        section .desc {{ color: #7f8c8d; margin-bottom: 1.2em; }}
        .plotly-graph-div {{ margin: 1em 0; }}
        .plot-container {{ margin: 1.5em 0; }}
        .toc {{
            background: white;
            padding: 1.5em 2em;
            border-radius: 8px;
            margin-bottom: 2em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .toc h3 {{ margin-bottom: 0.5em; color: #2c3e50; }}
        .toc a {{ color: #3498db; text-decoration: none; display: block; padding: 0.3em 0; font-size: 0.95em; }}
        .toc a:hover {{ text-decoration: underline; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        table td {{
            padding: 10px 14px;
            border-bottom: 1px solid #ecf0f1;
        }}
        table td:first-child {{ font-weight: 600; color: #34495e; width: 220px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }}
        .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1em; }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.2em;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .stat-card .value {{ font-size: 2em; font-weight: 700; color: #2c3e50; }}
        .stat-card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 0.3em; }}
        .badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.82em;
            font-weight: 600;
            margin: 2px;
        }}
        .badge-hbond {{ background: #d4efdf; color: #27ae60; }}
        .badge-hydrophobic {{ background: #fdebd0; color: #e67e22; }}
        .badge-ionic {{ background: #d6eaf8; color: #2980b9; }}
        .badge-pi {{ background: #e8daef; color: #8e44ad; }}
        .badge-water {{ background: #d1f2eb; color: #1abc9c; }}
        .insight-box {{
            background: linear-gradient(135deg, #ebf5fb, #d4efdf);
            border-left: 4px solid #27ae60;
            padding: 1em 1.5em;
            border-radius: 0 8px 8px 0;
            margin: 1em 0;
            font-size: 0.93em;
        }}
        .insight-box strong {{ color: #27ae60; }}
        .anomaly-box {{
            background: #fdf2e9;
            border-left: 4px solid #e67e22;
            padding: 1em 1.5em;
            border-radius: 0 8px 8px 0;
            margin: 1em 0;
        }}
        .anomaly-box strong {{ color: #e67e22; }}
        @media (max-width: 900px) {{
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
            body {{ padding: 1em; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | MDInsight v0.1.0</p>
    </div>

    <div class="toc">
        <h3>Table of Contents</h3>
        {toc_entries}
    </div>

    {"".join(sections_html)}

    <footer style="text-align:center;padding:2em;color:#7f8c8d;font-size:0.9em;">
        MDInsight &mdash; AI-Powered Deep Dive Analysis for Molecular Dynamics Simulations
    </footer>
</body>
</html>"""

        self.output_path.write_text(html)
        logger.info(f"Report saved: {self.output_path}")
        return self.output_path
