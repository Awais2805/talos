"""A self-contained page for reading a candidate's evidence.

Nothing here writes a verdict back -- that still means typing `analyst_class`
into the CSV (spreadsheet or otherwise); this page pairs with it as a reading
aid, plus a client-side CSV export so the two don't have to be edited by hand.
No network calls: opens from file://, survives an scp off a headless box.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from talos.data.labelling.audit.adjudication import COLUMNS, UNKNOWN

#: Columns adjudication.py always emits. Everything else in the CSV is flow.
_META = frozenset(COLUMNS)


class AuditPage:
    """Renders one dataset's audit CSV to a self-contained HTML file."""

    def __init__(self, space, dataset: str, method: str = "schedule"):
        self.space = space
        self.dataset = dataset
        self.method = method

    def render(self, duck, csv_path, html_path) -> Path:
        csv_path, html_path = Path(csv_path), Path(html_path)
        rel = duck.relation(
            f"SELECT * FROM read_csv_auto('{csv_path}', header = true, "
            f"all_varchar = true)")
        columns = list(rel.columns)
        flow = [c for c in columns if c not in _META]
        rows = [dict(zip(columns, row)) for row in duck.sql(rel.sql_query())]

        payload = {
            "dataset": self.dataset, "method": self.method, "columns": columns,
            "flow": flow, "classes": list(self.space.classes) + [UNKNOWN],
            "rows": rows,
        }
        html_path.parent.mkdir(parents=True, exist_ok=True)
        payload_json = json.dumps(payload).replace("<", "\\u003c")
        html_path.write_text(
            _PAGE.replace("__DATASET__", html.escape(self.dataset))
                 .replace("__METHOD__", html.escape(self.method))
                 .replace("__PAYLOAD__", payload_json))
        return html_path


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Audit — __DATASET__</title>
<style>
  body { font: 14px/1.4 -apple-system, sans-serif; margin: 1.5rem; color: #1a1a1a; }
  h1 { font-size: 1.1rem; margin-bottom: 0.25rem; }
  .method { color: #666; font-weight: normal; }
  #summary { color: #444; margin: 0 0 0.75rem; }
  button { padding: 0.4rem 0.9rem; cursor: pointer; }
  .tablewrap { overflow: auto; max-height: 80vh; border: 1px solid #ccc; margin-top: 0.75rem; }
  table { border-collapse: collapse; font: 12px/1.3 ui-monospace, monospace; white-space: nowrap; }
  th, td { padding: 0.3rem 0.5rem; border-bottom: 1px solid #eee; text-align: left; }
  thead th { position: sticky; top: 0; background: #f5f5f5; border-bottom: 1px solid #ccc; }
  tbody tr:hover { background: #fafafa; }
  select, input[type=text] { font: inherit; width: 100%; box-sizing: border-box; }
</style>
</head>
<body>
<h1>Audit — __DATASET__ <span class="method">(__METHOD__)</span></h1>
<p id="summary"></p>
<button id="export">Export adjudicated CSV</button>
<div class="tablewrap">
<table>
<thead><tr id="head-row"></tr></thead>
<tbody id="body-rows"></tbody>
</table>
</div>
<script type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.querySelector('script[type="application/json"]').textContent);

function buildTable() {
  const headRow = document.getElementById('head-row');
  for (const col of DATA.columns) {
    const th = document.createElement('th');
    th.textContent = col;
    headRow.appendChild(th);
  }
  const body = document.getElementById('body-rows');
  for (const row of DATA.rows) {
    const tr = document.createElement('tr');
    for (const col of DATA.columns) {
      const td = document.createElement('td');
      if (col === 'analyst_class') {
        const sel = document.createElement('select');
        sel.innerHTML = '<option value=""></option>' + DATA.classes
          .map(c => '<option value="' + c + '">' + c + '</option>').join('');
        sel.value = row[col] || '';
        sel.addEventListener('change', () => { row[col] = sel.value; updateSummary(); });
        td.appendChild(sel);
      } else if (col === 'analyst_note') {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.value = row[col] || '';
        inp.addEventListener('input', () => { row[col] = inp.value; });
        td.appendChild(inp);
      } else {
        td.textContent = row[col] == null ? '' : row[col];
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function updateSummary() {
  const done = DATA.rows.filter(r => r.analyst_class).length;
  document.getElementById('summary').textContent = done + ' / ' + DATA.rows.length + ' adjudicated';
}

function csvField(v) {
  const s = v == null ? '' : String(v);
  return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function exportCsv() {
  const lines = [DATA.columns.map(csvField).join(',')];
  for (const row of DATA.rows) lines.push(DATA.columns.map(c => csvField(row[c])).join(','));
  const blob = new Blob([lines.join('\\n')], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = DATA.dataset + '-adjudicated.csv';
  a.click();
}

buildTable();
updateSummary();
document.getElementById('export').addEventListener('click', exportCsv);
</script>
</body>
</html>
"""
