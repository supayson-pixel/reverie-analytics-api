<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Reverie Co — Lab</title>
  <style>
    :root {
      --bg:#0b1220; --panel:#101a2c; --ink:#eaf1ff; --dim:#9fb1d0;
      --brand1:#6fe1ff; --brand2:#6d7bff; --good:#19d39d; --bad:#ff6b6b;
      --border:#20304b; --accent:#192540;
    }
    * { box-sizing: border-box; }
    html, body { margin:0; padding:0; background:var(--bg); color:var(--ink);
      font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }
    a { color: var(--brand1); text-decoration: none; }
    .wrap { max-width: 1100px; margin: 42px auto; padding: 0 20px; }
    h1 { margin: 0 0 12px; font-size: 36px; }
    .crumb { margin-bottom: 20px; }
    .panel { background:linear-gradient(180deg, #121d34, #0e1727); border:1px solid var(--border); border-radius:16px; padding:20px; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    input[type=file] { padding: 10px; background: var(--panel); border:1px solid var(--border); color:var(--ink); border-radius:10px; }
    button {
      border:0; color:#081225; font-weight:700; padding:12px 18px; border-radius:12px; cursor:pointer;
      background:linear-gradient(90deg, var(--brand1), var(--brand2));
    }
    button.secondary { background:#1b2a46; color:var(--ink); border:1px solid var(--border); }
    .hint { color:var(--dim); font-size:14px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin-top:12px; }
    pre { white-space:pre-wrap; word-break:break-word; background: #0d1628; border:1px solid var(--border); padding:16px; border-radius:12px; }
    .badge { display:inline-block; padding:2px 8px; background:#14203a; border:1px solid var(--border); border-radius:100px; color:var(--dim); font-size:12px; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding:10px 8px; border-bottom:1px solid var(--border); text-align:left; }
    .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    .muted { color:var(--dim); }
    .good { color:var(--good); }
    .bad { color:var(--bad); }
    .section { margin-top:22px; }
    .sticky { position: sticky; top: 0; background: var(--bg); padding:8px 0 16px; z-index: 2; }
    .right { margin-left:auto; }
    code.copy { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; background:#0e1a2d; padding:4px 8px; border-radius:8px; border:1px solid var(--border); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="crumb"><a href="/">&larr; Back to site</a></div>
    <div class="sticky"><h1>Reverie Co — Lab</h1></div>

    <div class="panel">
      <div class="row" style="margin-bottom:10px">
        <div class="badge">API</div>
        <code class="copy" id="apiBase"></code>
        <span class="muted">Upload a CSV/XLSX to get a <code>dataset_id</code>, then profile it.</span>
      </div>

      <div class="controls">
        <input id="file" type="file" accept=".csv,.xlsx,.xls,.txt" />
        <button id="go">Upload &amp; Profile</button>
        <button class="secondary" id="copyJson" title="Copy JSON" disabled>Copy JSON</button>
        <button class="secondary" id="toggle" title="Show/Hide raw JSON" disabled>Toggle JSON</button>
        <span class="hint" id="status">Waiting for upload…</span>
      </div>

      <div class="section">
        <div class="grid">
          <div class="panel">
            <div class="muted">dataset_id</div>
            <div id="dsid" style="word-break: break-all">—</div>
          </div>
          <div class="panel">
            <div class="muted">Shape</div>
            <div id="shape">—</div>
          </div>
          <div class="panel">
            <div class="muted">Columns</div>
            <div id="columns">—</div>
          </div>
        </div>
      </div>

      <div class="section grid">
        <div class="panel">
          <div class="row"><strong>Nulls by column</strong></div>
          <div id="nulls">—</div>
        </div>
        <div class="panel">
          <div class="row"><strong>Top 5 categories</strong></div>
          <div class="muted" style="margin:-6px 0 8px">Date-like columns are excluded.</div>
          <div id="cats">—</div>
        </div>
      </div>

      <div class="section grid">
        <div class="panel">
          <div class="row"><strong>Numeric summary</strong></div>
          <div id="nums">—</div>
        </div>
        <div class="panel">
          <div class="row"><strong>Date summary</strong></div>
          <div class="muted" style="margin:-6px 0 8px">Min/Max/Span + by month & by weekday.</div>
          <div id="dates">—</div>
        </div>
      </div>

      <div class="section panel">
        <div class="row">
          <strong>Raw JSON</strong>
        </div>
        <pre id="jsonBox" hidden>{"status":"waiting"}</pre>
      </div>
    </div>
  </div>

  <script>
    // === Configure your live API here ===
    const API_URL = "https://reverie-analytics-api.onrender.com";
    // ====================================

    document.getElementById('apiBase').textContent = API_URL;

    const $ = sel => document.querySelector(sel);
    const statusEl = $('#status');
    const fileEl = $('#file');
    const goBtn = $('#go');
    const copyBtn = $('#copyJson');
    const toggleBtn = $('#toggle');
    const jsonBox = $('#jsonBox');

    let lastProfile = null;

    toggleBtn.addEventListener('click', () => {
      jsonBox.hidden = !jsonBox.hidden;
    });

    copyBtn.addEventListener('click', () => {
      if (!lastProfile) return;
      navigator.clipboard.writeText(JSON.stringify(lastProfile, null, 2));
      status('Copied JSON to clipboard ✔', 'good');
    });

    goBtn.addEventListener('click', async () => {
      try {
        const f = fileEl.files?.[0];
        if (!f) return status('Please choose a file first.', 'bad');
        if (f.size > 25 * 1024 * 1024) return status('File too large (limit ~25 MB for demo).', 'bad');

        status('Uploading… (cold starts can take a few seconds)');
        goBtn.disabled = true;

        const form = new FormData();
        form.append('file', f);
        const up = await fetch(`${API_URL}/analytics/upload`, { method:'POST', body: form });
        if (!up.ok) throw new Error(await up.text());
        const meta = await up.json();
        $('#dsid').textContent = meta.dataset_id || '—';

        status('Profiling…');
        const prof = await fetch(`${API_URL}/analytics/profile?dataset_id=${encodeURIComponent(meta.dataset_id)}`);
        if (!prof.ok) throw new Error(await prof.text());
        const data = await prof.json();
        lastProfile = data;
        copyBtn.disabled = false;
        toggleBtn.disabled = false;

        // Fill summary boxes
        $('#shape').textContent = data?.shape ? `${data.shape.rows} rows × ${data.shape.columns} cols` : '—';
        $('#columns').textContent = data?.columns?.length ? data.columns.join(', ') : '—';

        // Nulls
        $('#nulls').innerHTML = renderKVTable(data?.nulls || {}, 'Column', 'Nulls');

        // Top categories (already excludes dates on the API)
        $('#cats').innerHTML = Object.keys(data?.top5_categories || {}).length
          ? Object.entries(data.top5_categories).map(([col, kv]) => `
              <div style="margin:8px 0"><div class="muted">${escapeHtml(col)}</div>
                ${renderKVTable(kv, 'Value', 'Count')}
              </div>`).join('')
          : '—';

        // Numeric summary
        $('#nums').innerHTML = renderNumericSummary(data?.numeric_summary || {});

        // Date summary
        $('#dates').innerHTML = renderDateSummary(data?.date_summary || {});

        // Raw JSON
        jsonBox.textContent = JSON.stringify(data, null, 2);
        jsonBox.hidden = false;

        status('Done ✔', 'good');
      } catch (err) {
        console.error(err);
        status('Error: ' + (err?.message || err), 'bad');
      } finally {
        goBtn.disabled = false;
      }
    });

    function renderKVTable(obj, kLabel, vLabel) {
      const rows = Object.entries(obj);
      if (!rows.length) return '—';
      return `<table>
        <thead><tr><th>${kLabel}</th><th>${vLabel}</th></tr></thead>
        <tbody>${rows.map(([k,v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`).join('')}</tbody>
      </table>`;
    }

    function renderNumericSummary(numSummary) {
      const cols = Object.keys(numSummary || {});
      if (!cols.length) return '—';
      const wanted = ['count','mean','std','min','25%','50%','75%','max'];
      return cols.map(c => {
        const s = numSummary[c] || {};
        return `<div style="margin:8px 0">
          <div class="muted">${escapeHtml(c)}</div>
          <table>
            <thead><tr>${wanted.map(w => `<th>${w}</th>`).join('')}</tr></thead>
            <tbody>
              <tr>${wanted.map(w => `<td>${fmt(s[w])}</td>`).join('')}</tr>
            </tbody>
          </table>
        </div>`;
      }).join('');
    }

    function renderDateSummary(ds) {
      const cols = Object.keys(ds || {});
      if (!cols.length) return '—';
      return cols.map(col => {
        const d = ds[col] || {};
        const meta = [
          ['Min', d.min ?? '—'],
          ['Max', d.max ?? '—'],
          ['Span (days)', d.span_days ?? '—'],
        ];
        return `<div style="margin:8px 0">
          <div class="muted">${escapeHtml(col)}</div>
          <div class="grid">
            <div class="panel">
              <div class="muted">Range</div>
              ${renderList(meta)}
            </div>
            <div class="panel">
              <div class="muted">By month</div>
              ${renderKVTable(d.by_month || {}, 'Month', 'Count')}
            </div>
            <div class="panel">
              <div class="muted">By weekday</div>
              ${renderKVTable(d.by_weekday || {}, 'Weekday', 'Count')}
            </div>
          </div>
        </div>`;
      }).join('');
    }

    function renderList(pairs){
      return `<table>
        <tbody>${pairs.map(([k,v]) => `<tr><td class="muted" style="width:140px">${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join('')}</tbody>
      </table>`;
    }

    function fmt(x){
      return (x === null || x === undefined || Number.isNaN(x)) ? '—'
        : (typeof x === 'number' ? Number(x.toFixed(2)) : escapeHtml(String(x)));
    }
    function status(msg, mood){ statusEl.textContent = msg; statusEl.className = 'hint ' + (mood || ''); }
    function escapeHtml(s){ return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
  </script>
</body>
</html>
