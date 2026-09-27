/* Tabula engine: parsing, profiling, preprocessing, models, evaluation, tuning, forecasting, drift.
   Runs in the page, in a Web Worker (message protocol at the bottom), or in Node (module.exports). */
(function (root) {
  'use strict';
  const E = {};

  // ---------- utilities ----------
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    const f = function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    f.normal = function () { let u = 0, v = 0; while (u === 0) u = f(); v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    f.int = function (n) { return Math.floor(f() * n); };
    f.pick = function (arr) { return arr[Math.floor(f() * arr.length)]; };
    return f;
  }
  E.rng = rng;
  function shuffle(arr, r) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; } return arr; }
  function range(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }
  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : NaN; }
  function std(a, m) { if (m === undefined) m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return a.length > 1 ? Math.sqrt(s / (a.length - 1)) : 0; }
  function quantile(sorted, q) { if (!sorted.length) return NaN; const p = (sorted.length - 1) * q; const lo = Math.floor(p), hi = Math.ceil(p); return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo); }
  function sigmoid(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
  function softmaxInPlace(v) { let m = -Infinity; for (const x of v) if (x > m) m = x; let s = 0; for (let k = 0; k < v.length; k++) { v[k] = Math.exp(v[k] - m); s += v[k]; } for (let k = 0; k < v.length; k++) v[k] /= s; return v; }
  function round(x, d) { const p = Math.pow(10, d || 4); return Math.round(x * p) / p; }
  E.util = { mean, std, quantile, range, shuffle, round };

  // ---------- value parsing ----------
  const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '?', '-', '--', '#n/a', 'missing', 'undefined']);
  function isMissing(v) { if (v === null || v === undefined) return true; if (typeof v === 'number') return !isFinite(v); return MISSING.has(String(v).trim().toLowerCase()); }
  function parseNum(v) {
    if (typeof v === 'number') return isFinite(v) ? v : NaN;
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (isMissing(v)) return NaN;
    let s = String(v).trim();
    let neg = false;
    if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
    s = s.replace(/^[$\u20ac\xa3\xa5]\s?/, '').replace(/\s?[$\u20ac\xa3\xa5]$/, '').replace(/%$/, '');
    if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(s)) s = s.replace(/,/g, '');
    if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return NaN;
    const n = Number(s);
    return neg ? -n : n;
  }
  function parseDate(v) {
    if (v instanceof Date) return v.getTime();
    if (isMissing(v) || typeof v === 'number') return NaN;
    const s = String(v).trim();
    let m = s.match(/^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?/);
    if (m) {
      const y = +m[1], mo = +m[2], d = m[3] ? +m[3] : 1;
      if (mo < 1 || mo > 12 || d < 1 || d > 31) return NaN;
      return Date.UTC(y, mo - 1, d, m[4] ? +m[4] : 0, m[5] ? +m[5] : 0, m[6] ? +m[6] : 0);
    }
    m = s.match(/^(\d{1,2})[\/.](\d{1,2})[\/.](\d{2,4})(?:[ T](\d{1,2}):(\d{2}))?/);
    if (m) {
      let a = +m[1], b = +m[2], y = +m[3];
      if (y < 100) y += y < 50 ? 2000 : 1900;
      let mo = a, d = b; if (a > 12 && b <= 12) { mo = b; d = a; }
      if (mo < 1 || mo > 12 || d < 1 || d > 31) return NaN;
      return Date.UTC(y, mo - 1, d, m[4] ? +m[4] : 0, m[5] ? +m[5] : 0);
    }
    m = s.match(/^(\d{4})\/(\d{1,2})\/(\d{1,2})/);
    if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3]);
    return NaN;
  }
  E.parseNum = parseNum; E.parseDate = parseDate; E.isMissing = isMissing;

  // ---------- CSV ----------
  E.parseCSV = function (text) {
    text = String(text).replace(/^\ufeff/, '');
    const firstLine = text.slice(0, Math.min(text.length, 5000)).split(/\r?\n/)[0] || '';
    const cands = [',', ';', '\t', '|'];
    let delim = ',', best = -1;
    for (const c of cands) { const n = firstLine.split(c).length; if (n > best) { best = n; delim = c; } }
    const rows = []; let row = [], field = '', inQ = false, i = 0; const L = text.length;
    while (i < L) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; continue; } inQ = false; i++; continue; }
        field += ch; i++; continue;
      }
      if (ch === '"' && field === '') { inQ = true; i++; continue; }
      if (ch === delim) { row.push(field); field = ''; i++; continue; }
      if (ch === '\r') { i++; continue; }
      if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; i++; continue; }
      field += ch; i++;
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    const nonEmpty = rows.filter(r => r.length > 1 || (r[0] && r[0].trim() !== ''));
    if (!nonEmpty.length) throw new Error('The file has no rows.');
    let header = nonEmpty[0].map((h, j) => (h || '').trim() || ('column_' + (j + 1)));
    const seen = {};
    header = header.map(h => { if (seen[h]) { seen[h]++; return h + '_' + seen[h]; } seen[h] = 1; return h; });
    const out = [];
    for (let r = 1; r < nonEmpty.length; r++) {
      const o = {}; const src = nonEmpty[r];
      for (let j = 0; j < header.length; j++) o[header[j]] = src[j] !== undefined ? src[j].trim() : '';
      out.push(o);
    }
    return { columns: header, rows: out, delimiter: delim };
  };
  E.toCSV = function (columns, rows) {
    const esc = v => { if (v === null || v === undefined) return ''; const s = String(v); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    return [columns.map(esc).join(',')].concat(rows.map(r => columns.map(c => esc(r[c])).join(','))).join('\n');
  };
  // array of row objects (JSON, Parquet) -> table; converts BigInt, Date, boolean, nested values
  E.fromObjects = function (arr) {
    const cols = [], seen = new Set();
    for (const o of arr) for (const k in o) if (!seen.has(k)) { seen.add(k); cols.push(k); }
    const conv = v => {
      if (v === null || v === undefined) return '';
      if (typeof v === 'bigint') return Number(v);
      if (typeof v === 'number') return isFinite(v) ? v : '';
      if (v instanceof Date) { const t = v.getTime(); if (isNaN(t)) return ''; const iso = v.toISOString(); return t % 86400000 === 0 ? iso.slice(0, 10) : iso.slice(0, 19).replace('T', ' '); }
      if (typeof v === 'boolean') return v ? 'true' : 'false';
      if (typeof Uint8Array !== 'undefined' && v instanceof Uint8Array) return '';
      if (typeof v === 'object') return JSON.stringify(v, (k, x) => typeof x === 'bigint' ? Number(x) : x);
      return v;
    };
    return { columns: cols, rows: arr.map(o => { const r = {}; for (const c of cols) r[c] = conv(o[c]); return r; }) };
  };
  E.fromMatrix = function (matrix) { // array of arrays with header row (e.g. from a spreadsheet)
    const header = (matrix[0] || []).map((h, j) => (h === null || h === undefined || String(h).trim() === '') ? 'column_' + (j + 1) : String(h).trim());
    const rows = [];
    for (let r = 1; r < matrix.length; r++) {
      const src = matrix[r]; if (!src || src.every(v => v === null || v === undefined || v === '')) continue;
      const o = {}; header.forEach((h, j) => { let v = src[j]; if (v instanceof Date) v = v.toISOString().slice(0, 10); o[h] = v === undefined || v === null ? '' : v; });
      rows.push(o);
    }
    return { columns: header, rows };
  };

  // ---------- profiling ----------
  E.profile = function (table) {
    const n = table.rows.length;
    const cols = table.columns.map(name => {
      const raw = table.rows.map(r => r[name]);
      let missing = 0, numOk = 0, dateOk = 0, nonMiss = 0;
      const nums = [], dates = [], counts = new Map(); let totalLen = 0;
      for (const v of raw) {
        if (isMissing(v)) { missing++; continue; }
        nonMiss++;
        const s = typeof v === 'string' ? v.trim() : String(v);
        totalLen += s.length;
        counts.set(s, (counts.get(s) || 0) + 1);
        const x = parseNum(v); if (!isNaN(x)) { numOk++; nums.push(x); }
        const d = parseDate(v); if (!isNaN(d)) { dateOk++; dates.push(d); }
      }
      const unique = counts.size;
      const col = { name, missing, missingPct: n ? missing / n : 0, unique, n };
      const looksNumeric = nonMiss > 0 && numOk / nonMiss >= 0.95;
      const looksDate = nonMiss > 0 && dateOk / nonMiss >= 0.95 && !(looksNumeric && dates.length && !/[-\/]/.test(String(raw.find(v => !isMissing(v)))));
      const lname = name.toLowerCase();
      if (looksDate) {
        col.type = 'date'; dates.sort((a, b) => a - b);
        col.min = dates[0]; col.max = dates[dates.length - 1];
      } else if (looksNumeric) {
        nums.sort((a, b) => a - b);
        col.type = 'numeric';
        col.min = nums[0]; col.max = nums[nums.length - 1];
        col.mean = mean(nums); col.std = std(nums, col.mean); col.median = quantile(nums, 0.5);
        col.q1 = quantile(nums, 0.25); col.q3 = quantile(nums, 0.75);
        col.integer = nums.every(x => Number.isInteger(x));
        col.skew = col.std > 0 ? nums.reduce((s, x) => s + Math.pow((x - col.mean) / col.std, 3), 0) / nums.length : 0;
        col.hist = histogram(nums, 20);
        const idName = /(^|_|\b)(id|uuid|key|index|row)(_|\b|$)/i.test(name) || /id$/.test(name);
        if (col.integer && unique === nonMiss && nonMiss > 20 && idName) col.type = 'id';
      } else {
        const avgLen = nonMiss ? totalLen / nonMiss : 0;
        if (nonMiss > 20 && unique / nonMiss > 0.9) col.type = avgLen > 25 ? 'text' : 'id';
        else col.type = avgLen > 40 && unique / Math.max(1, nonMiss) > 0.5 ? 'text' : 'categorical';
        if (/(^|_)(id|uuid|key)$/i.test(lname) && unique / Math.max(1, nonMiss) > 0.5) col.type = 'id';
      }
      if (col.type === 'categorical' || col.type === 'id' || col.type === 'text' || (col.type === 'numeric' && unique <= 12)) {
        col.top = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([value, count]) => ({ value, count }));
      }
      col.constant = unique <= 1;
      col.sample = raw.filter(v => !isMissing(v)).slice(0, 3).map(String);
      col.role = (col.type === 'id' || col.type === 'text' || col.constant) ? 'ignore' : 'feature';
      col.reason = col.type === 'id' ? 'Looks like an identifier: every value is different, so it carries no pattern.' :
        col.type === 'text' ? 'Free text. This version does not learn from long text.' :
          col.constant ? 'Has only one value, so it cannot help a prediction.' : '';
      return col;
    });
    return { rows: n, columns: cols };
  };
  function histogram(sorted, bins) {
    const lo = sorted[0], hi = sorted[sorted.length - 1];
    if (!(hi > lo)) return { lo, hi, counts: [sorted.length] };
    const w = (hi - lo) / bins, counts = new Array(bins).fill(0);
    for (const x of sorted) { let b = Math.floor((x - lo) / w); if (b >= bins) b = bins - 1; counts[b]++; }
    return { lo, hi, counts };
  }

  E.suggestTask = function (profile, target) {
    const col = profile.columns.find(c => c.name === target);
    if (!col) return null;
    const hasDate = profile.columns.some(c => c.type === 'date' && c.name !== target);
    if (col.type === 'numeric') {
      if (col.unique <= 2) return { task: 'classification', why: 'The column has only two values, so it is treated as a yes/no outcome.', forecastable: hasDate };
      if (col.integer && col.unique <= 10 && profile.rows > 50) return { task: 'classification', why: 'Whole numbers with only ' + col.unique + ' distinct values look like categories.', alt: 'regression', forecastable: hasDate };
      return { task: 'regression', why: 'Numeric with many distinct values, so the model predicts a number.', forecastable: hasDate };
    }
    if (col.type === 'categorical' || col.type === 'id') return { task: 'classification', why: col.unique === 2 ? 'Two possible values, so this is a yes/no prediction.' : col.unique + ' possible values, so the model picks a category.', forecastable: false };
    return { task: 'classification', why: 'Treated as categories.', forecastable: false };
  };

  // ---------- preprocessing ----------
  // spec: [{name, type}] ; options: {maxCategories}
  E.fitPreprocessor = function (rows, features, opts) {
    opts = opts || {};
    const maxCat = opts.maxCategories || 30;
    const steps = [];
    for (const f of features) {
      if (f.type === 'numeric') {
        const vals = []; let miss = 0;
        for (const r of rows) { const x = parseNum(r[f.name]); if (isNaN(x)) miss++; else vals.push(x); }
        vals.sort((a, b) => a - b);
        const med = vals.length ? quantile(vals, 0.5) : 0;
        const m = vals.length ? mean(vals) : 0; const s = vals.length ? std(vals, m) || 1 : 1;
        const addInd = miss / Math.max(1, rows.length) > 0.02;
        steps.push({ name: f.name, type: 'numeric', impute: med, mean: m, std: s, indicator: addInd, cols: addInd ? [f.name, f.name + ' (missing)'] : [f.name] });
      } else if (f.type === 'categorical') {
        const counts = new Map();
        for (const r of rows) { const v = isMissing(r[f.name]) ? '(missing)' : String(r[f.name]).trim(); counts.set(v, (counts.get(v) || 0) + 1); }
        const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
        const minCount = rows.length > 200 ? 3 : 1;
        const cats = sorted.filter(([, c]) => c >= minCount).slice(0, maxCat).map(([v]) => v);
        const hasOther = cats.length < sorted.length;
        const levels = cats.concat(hasOther ? ['(other)'] : []);
        steps.push({ name: f.name, type: 'categorical', levels, mode: sorted.length ? sorted[0][0] : '', cols: levels.map(l => f.name + ' = ' + l) });
      } else if (f.type === 'date') {
        const ds = rows.map(r => parseDate(r[f.name])).filter(x => !isNaN(x));
        const hasTime = ds.some(d => d % 86400000 !== 0);
        const med = ds.length ? ds.slice().sort((a, b) => a - b)[Math.floor(ds.length / 2)] : 0;
        const parts = ['year', 'month', 'day of week', 'day of year'].concat(hasTime ? ['hour'] : []);
        const tmp = rows.map(r => dateParts(parseDate(r[f.name]), med, hasTime));
        const means = parts.map((_, k) => mean(tmp.map(t => t[k])));
        const stds = parts.map((_, k) => std(tmp.map(t => t[k]), means[k]) || 1);
        steps.push({ name: f.name, type: 'date', hasTime, impute: med, means, stds, cols: parts.map(p => f.name + ' \xb7 ' + p) });
      }
    }
    const names = []; const groups = [];
    for (const s of steps) { const start = names.length; for (const c of s.cols) names.push(c); groups.push({ feature: s.name, idx: range(s.cols.length).map(k => start + k) }); }
    return { steps, names, groups, dim: names.length };
  };
  function dateParts(t, impute, hasTime) {
    if (isNaN(t)) t = impute;
    const d = new Date(t);
    const start = Date.UTC(d.getUTCFullYear(), 0, 1);
    const out = [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDay(), Math.floor((t - start) / 86400000) + 1];
    if (hasTime) out.push(d.getUTCHours());
    return out;
  }
  E.transform = function (pp, rows) {
    const X = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i]; const x = new Float64Array(pp.dim); let p = 0;
      for (const s of pp.steps) {
        if (s.type === 'numeric') {
          let v = parseNum(r[s.name]); const miss = isNaN(v); if (miss) v = s.impute;
          x[p++] = (v - s.mean) / s.std;
          if (s.indicator) x[p++] = miss ? 1 : 0;
        } else if (s.type === 'categorical') {
          let v = isMissing(r[s.name]) ? '(missing)' : String(r[s.name]).trim();
          let k = s.levels.indexOf(v);
          if (k < 0) k = s.levels.indexOf('(other)');
          for (let j = 0; j < s.levels.length; j++) x[p + j] = j === k ? 1 : 0;
          p += s.levels.length;
        } else if (s.type === 'date') {
          const parts = dateParts(parseDate(r[s.name]), s.impute, s.hasTime);
          for (let k = 0; k < parts.length; k++) x[p++] = (parts[k] - s.means[k]) / s.stds[k];
        }
      }
      X[i] = x;
    }
    return X;
  };

  // target encoding
  E.encodeTarget = function (rows, target, task, classes) {
    if (task === 'classification') {
      if (!classes) {
        const counts = new Map();
        for (const r of rows) { if (isMissing(r[target])) continue; const v = String(r[target]).trim(); counts.set(v, (counts.get(v) || 0) + 1); }
        classes = Array.from(counts.keys()).sort((a, b) => {
          const na = parseNum(a), nb = parseNum(b);
          if (!isNaN(na) && !isNaN(nb)) return na - nb;
          const pos = /^(yes|true|y|1|positive|churn(ed)?)$/i;
          if (pos.test(a) && !pos.test(b)) return 1; if (pos.test(b) && !pos.test(a)) return -1;
          return a < b ? -1 : a > b ? 1 : 0;
        });
      }
      const y = rows.map(r => isMissing(r[target]) ? -1 : classes.indexOf(String(r[target]).trim()));
      return { y, classes };
    }
    return { y: rows.map(r => parseNum(r[target])) };
  };

  // ---------- binning for tree models ----------
  function makeBins(X, maxBins) {
    const n = X.length, d = n ? X[0].length : 0;
    const thresholds = new Array(d); const Xb = new Uint8Array(n * d);
    const col = new Float64Array(n);
    for (let j = 0; j < d; j++) {
      for (let i = 0; i < n; i++) col[i] = X[i][j];
      const sorted = Float64Array.from(col).sort();
      const uniq = []; for (let i = 0; i < n; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) uniq.push(sorted[i]);
      let th = [];
      if (uniq.length <= maxBins) { for (let k = 0; k < uniq.length - 1; k++) th.push((uniq[k] + uniq[k + 1]) / 2); }
      else {
        for (let k = 1; k < maxBins; k++) { const q = quantile(sorted, k / maxBins); th.push(q); }
        th = th.filter((t, k) => k === 0 || t > th[k - 1]);
        th = th.filter(t => t < sorted[n - 1]);
      }
      thresholds[j] = th;
      for (let i = 0; i < n; i++) {
        let lo = 0, hi = th.length; const v = col[i];
        while (lo < hi) { const mid = (lo + hi) >> 1; if (v <= th[mid]) hi = mid; else lo = mid + 1; }
        Xb[j * n + i] = lo;
      }
    }
    return { Xb, thresholds, n, d };
  }

  // multi-output gradient tree (covers CART regression, gini classification, and boosting)
  function buildTree(B, idx, G, H, K, p, r, imp) {
    const { Xb, thresholds, n, d } = B;
    const lambda = p.lambda || 0, minLeaf = p.minLeaf || 1, maxDepth = p.maxDepth || 6;
    const nodes = [];
    const nf = Math.max(1, Math.round(d * (p.featureFrac || 1)));
    const feats = range(d);
    function leafVal(Gs, Hs) { const v = new Array(K); for (let k = 0; k < K; k++) v[k] = -Gs[k] / (Hs[k] + lambda); return v; }
    function grow(ix, depth) {
      const m = ix.length; const Gs = new Float64Array(K), Hs = new Float64Array(K);
      for (let a = 0; a < m; a++) { const i = ix[a]; for (let k = 0; k < K; k++) { Gs[k] += G[i * K + k]; Hs[k] += H ? H[i * K + k] : 1; } }
      const id = nodes.length; nodes.push(null);
      if (depth >= maxDepth || m < 2 * minLeaf) { nodes[id] = { v: leafVal(Gs, Hs) }; return id; }
      let parentScore = 0; for (let k = 0; k < K; k++) parentScore += Gs[k] * Gs[k] / (Hs[k] + lambda);
      let bestGain = p.minGain || 1e-12, bestF = -1, bestB = -1;
      let fs = feats;
      if (nf < d) { fs = shuffle(feats.slice(), r).slice(0, nf); }
      const stride = 2 * K + 1;
      for (const j of fs) {
        const nb = thresholds[j].length + 1; if (nb < 2) continue;
        const hist = new Float64Array(nb * stride); const off = j * n;
        for (let a = 0; a < m; a++) {
          const i = ix[a]; const b = Xb[off + i] * stride;
          for (let k = 0; k < K; k++) { hist[b + k] += G[i * K + k]; hist[b + K + k] += H ? H[i * K + k] : 1; }
          hist[b + 2 * K] += 1;
        }
        const GL = new Float64Array(K), HL = new Float64Array(K); let cL = 0;
        for (let b = 0; b < nb - 1; b++) {
          const o = b * stride;
          for (let k = 0; k < K; k++) { GL[k] += hist[o + k]; HL[k] += hist[o + K + k]; }
          cL += hist[o + 2 * K];
          if (cL < minLeaf) continue; if (m - cL < minLeaf) break;
          let s = 0;
          for (let k = 0; k < K; k++) { const gr = Gs[k] - GL[k], hr = Hs[k] - HL[k]; s += GL[k] * GL[k] / (HL[k] + lambda) + gr * gr / (hr + lambda); }
          const gain = s - parentScore;
          if (gain > bestGain) { bestGain = gain; bestF = j; bestB = b; }
        }
      }
      if (bestF < 0) { nodes[id] = { v: leafVal(Gs, Hs) }; return id; }
      if (imp) imp[bestF] += bestGain;
      const L = [], R = []; const off = bestF * n;
      for (let a = 0; a < m; a++) { const i = ix[a]; if (Xb[off + i] <= bestB) L.push(i); else R.push(i); }
      const node = { f: bestF, t: thresholds[bestF][bestB], l: -1, r: -1 };
      nodes[id] = node;
      node.l = grow(L, depth + 1); node.r = grow(R, depth + 1);
      return id;
    }
    grow(idx, 0);
    return nodes;
  }
  function predictTree(nodes, x) { let k = 0; for (;;) { const nd = nodes[k]; if (nd.v) return nd.v; k = x[nd.f] <= nd.t ? nd.l : nd.r; } }

  // ---------- models ----------
  const MODELS = {};
  E.MODELS = MODELS;

  MODELS.baseline = {
    label: 'Baseline', short: 'Always predicts the average (or most common) value. Any useful model must beat it.',
    tasks: ['regression', 'classification'], space: {}, defaults: {},
    fit(X, y, task, K) {
      if (task === 'regression') return { c: mean(y) };
      const p = new Array(K).fill(0); for (const v of y) p[v]++; return { p: p.map(c => (c + 0.5) / (y.length + 0.5 * K)) };
    },
    predict(m, X, task) { return X.map(() => task === 'regression' ? m.c : m.p.slice()); }
  };

  MODELS.linear = {
    label: 'Linear model', short: 'Fits a straight-line relationship per feature. Fast, easy to explain, weak at interactions.',
    tasks: ['regression', 'classification'],
    defaults: { l2: 1, epochs: 60 },
    space: { l2: { type: 'log', min: 0.001, max: 100, label: 'Regularization (L2)' } },
    fit(X, y, task, K, p, r, onProg) {
      const n = X.length, d = n ? X[0].length : 0;
      if (task === 'regression') {
        const xm = new Float64Array(d); for (const x of X) for (let j = 0; j < d; j++) xm[j] += x[j] / n;
        const ym = mean(y);
        const A = new Float64Array(d * d), b = new Float64Array(d);
        for (let i = 0; i < n; i++) { const x = X[i], yi = y[i] - ym; for (let j = 0; j < d; j++) { const xj = x[j] - xm[j]; b[j] += xj * yi; for (let k = j; k < d; k++) A[j * d + k] += xj * (x[k] - xm[k]); } }
        for (let j = 0; j < d; j++) { for (let k = 0; k < j; k++) A[j * d + k] = A[k * d + j]; A[j * d + j] += p.l2 * Math.max(1, n / 100) + 1e-8; }
        const w = solveSPD(A, b, d);
        let c = ym; for (let j = 0; j < d; j++) c -= w[j] * xm[j];
        return { w: Array.from(w), c };
      }
      // softmax regression with Adam
      const W = new Float64Array(d * K), bb = new Float64Array(K);
      const mW = new Float64Array(d * K), vW = new Float64Array(d * K), mb = new Float64Array(K), vb = new Float64Array(K);
      const lr = 0.05, b1 = 0.9, b2 = 0.999; let t = 0;
      const order = range(n); const bs = 128; const epochs = p.epochs || 60; const reg = p.l2 / Math.max(n, 1);
      const gW = new Float64Array(d * K), gb = new Float64Array(K), z = new Float64Array(K);
      for (let ep = 0; ep < epochs; ep++) {
        shuffle(order, r);
        for (let s = 0; s < n; s += bs) {
          gW.fill(0); gb.fill(0); const e = Math.min(n, s + bs), m = e - s;
          for (let a = s; a < e; a++) {
            const i = order[a], x = X[i];
            for (let k = 0; k < K; k++) { let v = bb[k]; for (let j = 0; j < d; j++) v += x[j] * W[j * K + k]; z[k] = v; }
            softmaxInPlace(z);
            for (let k = 0; k < K; k++) { const g = z[k] - (y[i] === k ? 1 : 0); gb[k] += g; if (g !== 0) for (let j = 0; j < d; j++) gW[j * K + k] += g * x[j]; }
          }
          t++;
          const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
          for (let q = 0; q < d * K; q++) { const g = gW[q] / m + reg * W[q]; mW[q] = b1 * mW[q] + (1 - b1) * g; vW[q] = b2 * vW[q] + (1 - b2) * g * g; W[q] -= lr * (mW[q] / c1) / (Math.sqrt(vW[q] / c2) + 1e-8); }
          for (let k = 0; k < K; k++) { const g = gb[k] / m; mb[k] = b1 * mb[k] + (1 - b1) * g; vb[k] = b2 * vb[k] + (1 - b2) * g * g; bb[k] -= lr * (mb[k] / c1) / (Math.sqrt(vb[k] / c2) + 1e-8); }
        }
        if (onProg && ep % 10 === 0) onProg(ep / epochs);
      }
      return { W: Array.from(W), b: Array.from(bb), d, K };
    },
    predict(m, X, task) {
      if (task === 'regression') return X.map(x => { let v = m.c; for (let j = 0; j < m.w.length; j++) v += m.w[j] * x[j]; return v; });
      return X.map(x => { const z = new Array(m.K); for (let k = 0; k < m.K; k++) { let v = m.b[k]; for (let j = 0; j < m.d; j++) v += x[j] * m.W[j * m.K + k]; z[k] = v; } return softmaxInPlace(z); });
    }
  };
  function solveSPD(A, b, d) { // Cholesky with fallback
    const L = new Float64Array(d * d);
    for (let i = 0; i < d; i++) for (let j = 0; j <= i; j++) {
      let s = A[i * d + j]; for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
      if (i === j) L[i * d + i] = Math.sqrt(Math.max(s, 1e-10)); else L[i * d + j] = s / L[j * d + j];
    }
    const z = new Float64Array(d);
    for (let i = 0; i < d; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * d + k] * z[k]; z[i] = s / L[i * d + i]; }
    const w = new Float64Array(d);
    for (let i = d - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < d; k++) s -= L[k * d + i] * w[k]; w[i] = s / L[i * d + i]; }
    return w;
  }

  function treeTargets(y, task, K, n) {
    const G = new Float64Array(n * K);
    for (let i = 0; i < n; i++) { if (task === 'regression') G[i] = -y[i]; else G[i * K + y[i]] = -1; }
    return G;
  }
  MODELS.tree = {
    label: 'Decision tree', short: 'A single flowchart of yes/no questions. Very readable, but tends to memorize noise.',
    tasks: ['regression', 'classification'],
    defaults: { maxDepth: 6, minLeaf: 10 },
    space: { maxDepth: { type: 'int', min: 2, max: 14, label: 'Max depth' }, minLeaf: { type: 'int', min: 1, max: 60, label: 'Min rows per leaf' } },
    fit(X, y, task, K, p) {
      const Ko = task === 'regression' ? 1 : K; const n = X.length;
      const B = makeBins(X, 64); const imp = new Float64Array(B.d);
      const nodes = buildTree(B, range(n), treeTargets(y, task, Ko, n), null, Ko, { maxDepth: p.maxDepth, minLeaf: p.minLeaf, lambda: 0 }, null, imp);
      return { nodes, imp: Array.from(imp) };
    },
    predict(m, X, task) { return X.map(x => { const v = predictTree(m.nodes, x); return task === 'regression' ? v[0] : v.slice(); }); }
  };

  MODELS.forest = {
    label: 'Random forest', short: 'Averages many trees, each trained on a random slice of rows and columns. Strong, stable default.',
    tasks: ['regression', 'classification'],
    defaults: { trees: 80, maxDepth: 12, minLeaf: 3, featureFrac: 0.5 },
    space: { trees: { type: 'int', min: 30, max: 200, label: 'Number of trees' }, maxDepth: { type: 'int', min: 4, max: 20, label: 'Max depth' }, minLeaf: { type: 'int', min: 1, max: 30, label: 'Min rows per leaf' }, featureFrac: { type: 'float', min: 0.15, max: 1, label: 'Share of columns per split' } },
    fit(X, y, task, K, p, r, onProg) {
      const Ko = task === 'regression' ? 1 : K; const n = X.length;
      const B = makeBins(X, 48); const G = treeTargets(y, task, Ko, n); const imp = new Float64Array(B.d);
      const trees = [];
      for (let t = 0; t < p.trees; t++) {
        const idx = new Array(n); for (let i = 0; i < n; i++) idx[i] = Math.floor(r() * n);
        trees.push(buildTree(B, idx, G, null, Ko, { maxDepth: p.maxDepth, minLeaf: p.minLeaf, lambda: 0, featureFrac: p.featureFrac }, r, imp));
        if (onProg && t % 5 === 0) onProg(t / p.trees);
      }
      return { trees, imp: Array.from(imp) };
    },
    predict(m, X, task) {
      return X.map(x => {
        const K = m.trees[0][0].v ? m.trees[0][0].v.length : null;
        let acc = null;
        for (const t of m.trees) { const v = predictTree(t, x); if (!acc) acc = v.slice(); else for (let k = 0; k < v.length; k++) acc[k] += v[k]; }
        for (let k = 0; k < acc.length; k++) acc[k] /= m.trees.length;
        return task === 'regression' ? acc[0] : acc;
      });
    }
  };

  MODELS.boosting = {
    label: 'Gradient boosting', short: 'Builds small trees one after another, each fixing the last one\'s mistakes. Often the most accurate on tables.',
    tasks: ['regression', 'classification'],
    defaults: { rounds: 150, learningRate: 0.08, maxDepth: 4, minLeaf: 10, subsample: 0.8, featureFrac: 0.9, lambda: 1 },
    space: { rounds: { type: 'int', min: 50, max: 400, label: 'Boosting rounds' }, learningRate: { type: 'log', min: 0.02, max: 0.3, label: 'Learning rate' }, maxDepth: { type: 'int', min: 2, max: 8, label: 'Tree depth' }, minLeaf: { type: 'int', min: 2, max: 50, label: 'Min rows per leaf' }, subsample: { type: 'float', min: 0.5, max: 1, label: 'Row sample per round' }, lambda: { type: 'log', min: 0.1, max: 20, label: 'Leaf regularization' } },
    fit(X, y, task, K, p, r, onProg) {
      const n = X.length; const B = makeBins(X, 48); const imp = new Float64Array(B.d);
      const Ko = task === 'regression' ? 1 : (K === 2 ? 1 : K);
      const F = new Float64Array(n * Ko); let init;
      if (task === 'regression') init = [mean(y)];
      else if (Ko === 1) { const p1 = Math.min(0.999, Math.max(0.001, y.reduce((s, v) => s + v, 0) / n)); init = [Math.log(p1 / (1 - p1))]; }
      else { const c = new Array(K).fill(1); for (const v of y) c[v]++; init = c.map(x => Math.log(x / (n + K))); }
      for (let i = 0; i < n; i++) for (let k = 0; k < Ko; k++) F[i * Ko + k] = init[k];
      const G = new Float64Array(n * Ko), H = new Float64Array(n * Ko); const trees = [];
      const pr = new Array(Ko);
      for (let t = 0; t < p.rounds; t++) {
        for (let i = 0; i < n; i++) {
          if (task === 'regression') { G[i] = F[i] - y[i]; H[i] = 1; }
          else if (Ko === 1) { const q = sigmoid(F[i]); G[i] = q - y[i]; H[i] = Math.max(q * (1 - q), 1e-6); }
          else { for (let k = 0; k < Ko; k++) pr[k] = F[i * Ko + k]; softmaxInPlace(pr); for (let k = 0; k < Ko; k++) { G[i * Ko + k] = pr[k] - (y[i] === k ? 1 : 0); H[i * Ko + k] = Math.max(pr[k] * (1 - pr[k]), 1e-6); } }
        }
        let idx = range(n); if (p.subsample < 1) idx = idx.filter(() => r() < p.subsample);
        const tree = buildTree(B, idx, G, H, Ko, { maxDepth: p.maxDepth, minLeaf: p.minLeaf, lambda: p.lambda, featureFrac: p.featureFrac || 1 }, r, imp);
        for (const nd of tree) if (nd.v) for (let k = 0; k < Ko; k++) nd.v[k] *= p.learningRate;
        trees.push(tree);
        for (let i = 0; i < n; i++) { const v = predictTree(tree, X[i]); for (let k = 0; k < Ko; k++) F[i * Ko + k] += v[k]; }
        if (onProg && t % 10 === 0) onProg(t / p.rounds);
      }
      return { trees, init, Ko, imp: Array.from(imp) };
    },
    predict(m, X, task) {
      return X.map(x => {
        const f = m.init.slice();
        for (const t of m.trees) { const v = predictTree(t, x); for (let k = 0; k < m.Ko; k++) f[k] += v[k]; }
        if (task === 'regression') return f[0];
        if (m.Ko === 1) { const q = sigmoid(f[0]); return [1 - q, q]; }
        return softmaxInPlace(f);
      });
    }
  };

  MODELS.knn = {
    label: 'Nearest neighbors', short: 'Predicts from the most similar rows it has seen. Simple, but slow and sensitive to irrelevant columns.',
    tasks: ['regression', 'classification'],
    defaults: { k: 15 },
    space: { k: { type: 'int', min: 3, max: 60, label: 'Neighbors (k)' } },
    fit(X, y, task, K, p, r) {
      let idx = range(X.length); if (idx.length > 2500) idx = shuffle(idx, r).slice(0, 2500);
      return { X: idx.map(i => Array.from(X[i])), y: idx.map(i => y[i]), k: p.k, K };
    },
    predict(m, X, task) {
      const k = Math.min(m.k, m.X.length);
      return X.map(x => {
        const best = []; // [dist, idx] sorted ascending, size k
        for (let i = 0; i < m.X.length; i++) {
          const z = m.X[i]; let dd = 0; for (let j = 0; j < z.length; j++) { const q = z[j] - x[j]; dd += q * q; }
          if (best.length < k) { best.push([dd, i]); best.sort((a, b) => a[0] - b[0]); }
          else if (dd < best[k - 1][0]) { best[k - 1] = [dd, i]; best.sort((a, b) => a[0] - b[0]); }
        }
        let wsum = 0;
        if (task === 'regression') { let s = 0; for (const [dd, i] of best) { const w = 1 / (Math.sqrt(dd) + 1e-6); s += w * m.y[i]; wsum += w; } return s / wsum; }
        const pr = new Array(m.K).fill(0.01); for (const [dd, i] of best) { const w = 1 / (Math.sqrt(dd) + 1e-3); pr[m.y[i]] += w; wsum += w; }
        const tot = pr.reduce((a, b) => a + b, 0); return pr.map(v => v / tot);
      });
    }
  };

  // ---------- metrics ----------
  const METRICS = {
    rmse: { label: 'RMSE', better: 'lower', help: 'Typical size of an error, in the target\'s own units. Punishes big misses.' },
    mae: { label: 'MAE', better: 'lower', help: 'Average absolute error, in the target\'s own units.' },
    r2: { label: 'R\xb2', better: 'higher', help: 'Share of the variation the model explains. 1 is perfect, 0 is no better than the average.' },
    mape: { label: 'MAPE', better: 'lower', help: 'Average error as a percent of the true value.' },
    accuracy: { label: 'Accuracy', better: 'higher', help: 'Share of rows predicted correctly.' },
    balanced_accuracy: { label: 'Balanced accuracy', better: 'higher', help: 'Accuracy averaged over classes, so rare classes count as much as common ones.' },
    macro_f1: { label: 'Macro F1', better: 'higher', help: 'Balance of precision and recall, averaged over classes.' },
    auc: { label: 'AUC', better: 'higher', help: 'How well the model ranks positives above negatives. 0.5 is a coin flip, 1 is perfect.' },
    logloss: { label: 'Log loss', better: 'lower', help: 'Penalizes confident wrong probabilities. Lower is better.' },
    smape: { label: 'sMAPE', better: 'lower', help: 'Symmetric percent error, 0 to 200%.' },
  };
  E.METRICS = METRICS;
  E.regressionMetrics = function (y, p) {
    const n = y.length; let se = 0, ae = 0, ape = 0, apeN = 0, sp = 0;
    const ym = mean(y); let ss = 0;
    for (let i = 0; i < n; i++) { const e = p[i] - y[i]; se += e * e; ae += Math.abs(e); ss += (y[i] - ym) * (y[i] - ym); if (Math.abs(y[i]) > 1e-9) { ape += Math.abs(e / y[i]); apeN++; } const den = Math.abs(y[i]) + Math.abs(p[i]); if (den > 0) sp += 2 * Math.abs(e) / den; }
    return { rmse: Math.sqrt(se / n), mae: ae / n, r2: ss > 0 ? 1 - se / ss : 0, mape: apeN ? 100 * ape / apeN : NaN, smape: 100 * sp / n };
  };
  E.classificationMetrics = function (y, P, K, threshold) {
    const n = y.length;
    const pred = P.map(p => (K === 2 && threshold !== undefined) ? (p[1] >= threshold ? 1 : 0) : argmax(p));
    const cm = Array.from({ length: K }, () => new Array(K).fill(0));
    let correct = 0, ll = 0;
    for (let i = 0; i < n; i++) { cm[y[i]][pred[i]]++; if (pred[i] === y[i]) correct++; ll -= Math.log(Math.max(1e-15, P[i][y[i]])); }
    const f1s = [], recalls = [];
    for (let k = 0; k < K; k++) {
      const tp = cm[k][k]; let fp = 0, fn = 0; for (let j = 0; j < K; j++) { if (j !== k) { fp += cm[j][k]; fn += cm[k][j]; } }
      const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
      recalls.push(rec); f1s.push(prec + rec ? 2 * prec * rec / (prec + rec) : 0);
    }
    let auc;
    if (K === 2) auc = aucBinary(y, P.map(p => p[1]));
    else { const a = []; for (let k = 0; k < K; k++) { const yk = y.map(v => v === k ? 1 : 0); if (yk.some(v => v) && yk.some(v => !v)) a.push(aucBinary(yk, P.map(p => p[k]))); } auc = a.length ? mean(a) : NaN; }
    const out = { accuracy: correct / n, balanced_accuracy: mean(recalls), macro_f1: mean(f1s), auc, logloss: ll / n, confusion: cm };
    if (K === 2) { const tp = cm[1][1], fp = cm[0][1], fn = cm[1][0]; out.precision = tp + fp ? tp / (tp + fp) : 0; out.recall = tp + fn ? tp / (tp + fn) : 0; }
    return out;
  };
  function argmax(v) { let b = 0; for (let k = 1; k < v.length; k++) if (v[k] > v[b]) b = k; return b; }
  function aucBinary(y, s) {
    const idx = range(y.length).sort((a, b) => s[a] - s[b]);
    let rank = 1, sumPos = 0, nPos = 0, nNeg = 0;
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
      const avg = (rank + rank + (j - i)) / 2;
      for (let q = i; q <= j; q++) { if (y[idx[q]] === 1) { sumPos += avg; nPos++; } else nNeg++; }
      rank += j - i + 1; i = j + 1;
    }
    return nPos && nNeg ? (sumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg) : NaN;
  }
  E.defaultMetric = function (task, classCounts) {
    if (task === 'regression' || task === 'forecast') return task === 'forecast' ? 'mae' : 'rmse';
    const tot = classCounts.reduce((a, b) => a + b, 0); const minShare = Math.min.apply(null, classCounts) / tot;
    if (classCounts.length === 2) return minShare < 0.25 ? 'auc' : 'accuracy';
    return minShare < 0.1 ? 'macro_f1' : 'accuracy';
  };
  function better(metric, a, b) { return METRICS[metric].better === 'higher' ? a > b : a < b; }
  E.better = better;

  // ---------- splits ----------
  function splitIndices(n, y, task, validFrac, r, orderBy) {
    if (orderBy) { const idx = range(n).sort((a, b) => orderBy[a] - orderBy[b]); const cut = Math.floor(n * (1 - validFrac)); return { train: idx.slice(0, cut), valid: idx.slice(cut) }; }
    if (task === 'classification') {
      const byClass = {}; for (let i = 0; i < n; i++) (byClass[y[i]] = byClass[y[i]] || []).push(i);
      const train = [], valid = [];
      for (const k in byClass) { const ix = shuffle(byClass[k], r); const c = Math.round(ix.length * validFrac); valid.push(...ix.slice(0, c)); train.push(...ix.slice(c)); }
      return { train: shuffle(train, r), valid: shuffle(valid, r) };
    }
    const idx = shuffle(range(n), r); const c = Math.round(n * validFrac);
    return { train: idx.slice(c), valid: idx.slice(0, c) };
  }
  function kfold(n, k, r) { const idx = shuffle(range(n), r); const folds = []; for (let f = 0; f < k; f++) folds.push(idx.filter((_, i) => i % k === f)); return folds; }

  // ---------- tabular experiment ----------
  // setup: {target, task, features:[{name,type}], validFrac, seed, algos:[names], params:{algo:{..}}, metric, maxRows, orderBy}
  E.prepareData = function (table, setup) {
    const rows = table.rows.filter(r => !isMissing(r[setup.target]) && (setup.task !== 'regression' || !isNaN(parseNum(r[setup.target]))));
    const enc = E.encodeTarget(rows, setup.target, setup.task, setup.classes);
    let keep = range(rows.length).filter(i => setup.task !== 'classification' || enc.y[i] >= 0);
    const r = rng(setup.seed || 7);
    if (setup.maxRows && keep.length > setup.maxRows) keep = shuffle(keep, r).slice(0, setup.maxRows);
    const R = keep.map(i => rows[i]); const y = keep.map(i => enc.y[i]);
    return { rows: R, y, classes: enc.classes, dropped: table.rows.length - rows.length };
  };

  E.runExperiment = function (table, setup, onProgress) {
    const prog = onProgress || function () {};
    const data = E.prepareData(table, setup);
    const { rows, y, classes } = data; const task = setup.task; const K = classes ? classes.length : 1;
    if (rows.length < 20) throw new Error('Only ' + rows.length + ' usable rows. At least 20 rows with a target value are needed.');
    if (task === 'classification' && K < 2) throw new Error('The target has only one value, so there is nothing to predict.');
    const r = rng(setup.seed || 7);
    const orderBy = setup.orderBy ? rows.map(row => { const d = parseDate(row[setup.orderBy]); return isNaN(d) ? parseNum(row[setup.orderBy]) : d; }) : null;
    const sp = splitIndices(rows.length, y, task, setup.validFrac || 0.2, r, orderBy);
    const trRows = sp.train.map(i => rows[i]), vaRows = sp.valid.map(i => rows[i]);
    const ytr = sp.train.map(i => y[i]), yva = sp.valid.map(i => y[i]);
    const pp = E.fitPreprocessor(trRows, setup.features);
    if (!pp.dim) throw new Error('No feature columns are selected. Include at least one column in step 3.');
    const Xtr = E.transform(pp, trRows), Xva = E.transform(pp, vaRows);
    const classCounts = classes ? classes.map((_, k) => y.filter(v => v === k).length) : null;
    const metric = setup.metric || E.defaultMetric(task, classCounts || []);
    const results = [];
    const algos = setup.algos && setup.algos.length ? setup.algos : ['baseline', 'linear', 'forest', 'boosting'];
    algos.forEach((name, ai) => {
      const M = MODELS[name]; const params = Object.assign({}, M.defaults, (setup.params || {})[name] || {});
      const t0 = Date.now();
      const model = M.fit(Xtr, ytr, task, K, params, rng((setup.seed || 7) + ai), f => prog({ stage: 'train', algo: name, i: ai, of: algos.length, frac: f }));
      const ms = Date.now() - t0;
      prog({ stage: 'train', algo: name, i: ai + 1, of: algos.length, frac: 0 });
      const pva = M.predict(model, Xva, task), ptr = M.predict(model, Xtr.slice(0, 3000), task);
      const mv = task === 'regression' ? E.regressionMetrics(yva, pva) : E.classificationMetrics(yva, pva, K);
      const mt = task === 'regression' ? E.regressionMetrics(ytr.slice(0, 3000), ptr) : E.classificationMetrics(ytr.slice(0, 3000), ptr, K);
      results.push({ algo: name, label: M.label, params, valid: stripConf(mv), train: stripConf(mt), confusion: mv.confusion, ms, model, score: mv[metric] });
    });
    results.forEach(x => x.key = x.algo);
    const scored = results.filter(x => !isNaN(x.score));
    scored.sort((a, b) => better(metric, a.score, b.score) ? -1 : 1);
    const best = scored[0] || results[0];
    LAST = { setup, task, K, metric, pp, Xtr, ytr, Xva, yva, classCounts, data, rows: {} };
    results.forEach(x => LAST.rows[x.key] = x);
    return {
      task, metric, classes, classCounts, nTrain: trRows.length, nValid: vaRows.length, dropped: data.dropped,
      features: pp.steps.map(s => s.name),
      results: results.map(publicRow),
      bestKey: best.key, inspect: E.inspect(best.key)
    };
  };
  let LAST = null;
  function publicRow(x) { return { key: x.key, algo: x.algo, label: x.label, params: x.params, valid: x.valid, train: x.train, ms: x.ms, score: x.score, tuned: !!x.tuned }; }
  E.inspect = function (key) {
    if (!LAST || !LAST.rows[key]) throw new Error('Those training results are no longer in memory. Train again.');
    const L = LAST, row = L.rows[key], M = MODELS[row.algo];
    const P = M.predict(row.model, L.Xva, L.task);
    const imp = row.algo === 'baseline' ? [] : permutationImportance(M, row.model, L.Xva, L.yva, L.task, L.K, L.pp.groups, L.metric, rng(99));
    const sample = []; const step = Math.max(1, Math.floor(L.Xva.length / 400));
    for (let i = 0; i < L.Xva.length; i += step) sample.push({ actual: L.yva[i], pred: P[i] });
    const results = Object.values(L.rows);
    return {
      key, importance: imp, sample,
      probs: L.task === 'classification' ? P.map((p, i) => [L.yva[i], L.K === 2 ? p[1] : p]) : null,
      confusion: row.confusion,
      diagnostics: diagnostics(L.setup, L.data, L.pp, results, row, imp, L.metric, L.classCounts)
    };
  };
  E.trainCandidate = function (algo, params, onProgress) {
    if (!LAST) throw new Error('Train the models first.');
    const L = LAST, M = MODELS[algo]; const p = Object.assign({}, M.defaults, params || {});
    const t0 = Date.now();
    const model = M.fit(L.Xtr, L.ytr, L.task, L.K, p, rng((L.setup.seed || 7) + 77), onProgress ? f => onProgress({ stage: 'train', algo, i: 0, of: 1, frac: f }) : null);
    const ms = Date.now() - t0;
    const pva = M.predict(model, L.Xva, L.task), ptr = M.predict(model, L.Xtr.slice(0, 3000), L.task);
    const mv = L.task === 'regression' ? E.regressionMetrics(L.yva, pva) : E.classificationMetrics(L.yva, pva, L.K);
    const mt = L.task === 'regression' ? E.regressionMetrics(L.ytr.slice(0, 3000), ptr) : E.classificationMetrics(L.ytr.slice(0, 3000), ptr, L.K);
    let n = 1; while (L.rows[algo + '#' + n]) n++;
    const key = algo + '#' + n;
    const row = { key, algo, label: M.label + ' \xb7 tuned ' + n, params: p, valid: stripConf(mv), train: stripConf(mt), confusion: mv.confusion, ms, model, score: mv[L.metric], tuned: true };
    L.rows[key] = row;
    return { row: publicRow(row), inspect: E.inspect(key) };
  };
  E.thresholdMetrics = function (probs, threshold) {
    return E.classificationMetrics(probs.map(q => q[0]), probs.map(q => [1 - q[1], q[1]]), 2, threshold);
  };
  function stripConf(m) { const o = Object.assign({}, m); delete o.confusion; return o; }

  function permutationImportance(M, model, X, y, task, K, groups, metric, r) {
    let idx = range(X.length); if (idx.length > 1500) idx = shuffle(idx, r).slice(0, 1500);
    const Xs = idx.map(i => X[i]), ys = idx.map(i => y[i]);
    const score = P => (task === 'regression' ? E.regressionMetrics(ys, P) : E.classificationMetrics(ys, P, K))[metric];
    const base = score(M.predict(model, Xs, task));
    const out = [];
    for (const g of groups) {
      const perm = shuffle(range(Xs.length), r);
      const Xp = Xs.map((x, i) => { const z = Float64Array.from(x); for (const j of g.idx) z[j] = Xs[perm[i]][j]; return z; });
      const s = score(M.predict(model, Xp, task));
      const drop = METRICS[metric].better === 'higher' ? base - s : s - base;
      out.push({ feature: g.feature, drop });
    }
    const tot = out.reduce((a, b) => a + Math.max(0, b.drop), 0) || 1;
    out.forEach(o => o.share = Math.max(0, o.drop) / tot);
    return out.sort((a, b) => b.drop - a.drop);
  }

  function diagnostics(setup, data, pp, results, best, imp, metric, classCounts) {
    const notes = [];
    const base = results.find(x => x.algo === 'baseline');
    if (base && best.algo !== 'baseline') {
      const gain = METRICS[metric].better === 'higher' ? best.score - base.score : base.score - best.score;
      if (Math.abs(gain) < 1e-9 || gain <= 0) notes.push({ level: 'critical', title: 'No model beats the baseline', text: 'The features may not carry information about the target, or there are too few rows. Check the columns you included.' });
    } else if (best.algo === 'baseline') {
      const others = results.filter(x => x.algo !== 'baseline' && !isNaN(x.score));
      if (others.length && !others.some(x => better(metric, x.score, best.score))) notes.push({ level: 'critical', title: 'The baseline won', text: 'None of the models learned a useful pattern. Try including more columns or collecting more rows.' });
      else notes.push({ level: 'info', title: 'This is the baseline', text: 'It always guesses the ' + (setup.task === 'regression' ? 'average' : 'most common value') + '. Use it as the bar the other models must clear.' });
      return notes;
    }
    if (imp.length && imp[0].share > 0.7 && imp.length > 2) {
      const hi = setup.task === 'regression' ? best.valid.r2 > 0.97 : best.valid.accuracy > 0.98;
      notes.push({ level: hi ? 'serious' : 'warning', feature: imp[0].feature, title: (hi ? 'Possible leakage: ' : 'One column dominates: ') + imp[0].feature, text: hi ? 'Scores are near perfect and rely almost entirely on this column. If it is only known after the outcome, exclude it or the model will fail in real use.' : 'Most of the predictive power comes from this one column. Make sure it will be available when you need predictions.' });
    }
    const tr = best.train[metric], va = best.valid[metric];
    if (setup.task === 'regression' ? (best.train.r2 - best.valid.r2 > 0.15) : (best.train.accuracy - best.valid.accuracy > 0.1)) {
      notes.push({ level: 'warning', title: 'Overfitting', text: 'The model does much better on rows it trained on than on held-out rows. Tuning with more regularization (shallower trees, larger leaves) usually helps.' });
    }
    if (classCounts) {
      const tot = classCounts.reduce((a, b) => a + b, 0); const mn = Math.min.apply(null, classCounts);
      if (mn / tot < 0.15) notes.push({ level: 'warning', title: 'Imbalanced classes', text: 'The rarest class is only ' + Math.round(100 * mn / tot) + '% of rows. Accuracy can look good while missing it; watch AUC, balanced accuracy and the confusion matrix.' });
    }
    if (data.rows.length < 200) notes.push({ level: 'warning', title: 'Small dataset', text: 'With under 200 rows, scores can swing a lot between runs. Treat differences between models with caution.' });
    const weak = imp.filter(o => o.drop <= 0).map(o => o.feature);
    if (weak.length && weak.length < imp.length) notes.push({ level: 'info', title: weak.length + ' column' + (weak.length > 1 ? 's add' : ' adds') + ' nothing', text: 'Shuffling ' + weak.slice(0, 5).join(', ') + (weak.length > 5 ? '\u2026' : '') + ' did not hurt the score. Removing them can make the model simpler without losing accuracy.', features: weak });
    return notes;
  }

  // ---------- tuning ----------
  E.tune = function (table, setup, algo, trials, onProgress) {
    const prog = onProgress || function () {};
    const data = E.prepareData(table, Object.assign({}, setup, { maxRows: Math.min(setup.maxRows || 20000, 8000) }));
    const { rows, y, classes } = data; const task = setup.task; const K = classes ? classes.length : 1;
    const M = MODELS[algo]; const r = rng((setup.seed || 7) + 1000);
    const pp = E.fitPreprocessor(rows, setup.features); const X = E.transform(pp, rows);
    const classCounts = classes ? classes.map((_, k) => y.filter(v => v === k).length) : null;
    const metric = setup.metric || E.defaultMetric(task, classCounts || []);
    const folds = kfold(rows.length, 3, r);
    const space = M.space; const out = [];
    const cand = [Object.assign({}, M.defaults, (setup.params || {})[algo] || {})];
    for (let t = 1; t < trials; t++) {
      const p = Object.assign({}, cand[0]);
      for (const k in space) { const s = space[k]; if (s.type === 'int') p[k] = s.min + Math.floor(r() * (s.max - s.min + 1)); else if (s.type === 'log') p[k] = round(Math.exp(Math.log(s.min) + r() * (Math.log(s.max) - Math.log(s.min))), 4); else p[k] = round(s.min + r() * (s.max - s.min), 3); }
      cand.push(p);
    }
    cand.forEach((p, t) => {
      const scores = [];
      folds.forEach((vf, f) => {
        const inV = new Uint8Array(rows.length); vf.forEach(i => inV[i] = 1);
        const tr = range(rows.length).filter(i => !inV[i]);
        const model = M.fit(tr.map(i => X[i]), tr.map(i => y[i]), task, K, p, rng(t * 7 + f));
        const P = M.predict(model, vf.map(i => X[i]), task); const yv = vf.map(i => y[i]);
        const m = task === 'regression' ? E.regressionMetrics(yv, P) : E.classificationMetrics(yv, P, K);
        scores.push(m[metric]);
      });
      out.push({ params: p, score: mean(scores), spread: std(scores), isCurrent: t === 0 });
      prog({ stage: 'tune', i: t + 1, of: cand.length });
    });
    out.sort((a, b) => better(metric, a.score, b.score) ? -1 : 1);
    return { algo, metric, trials: out, best: out[0].params };
  };

  // ---------- final fit + prediction ----------
  E.fitFinal = function (table, setup, algo, params, onProgress) {
    const data = E.prepareData(table, setup);
    const pp = E.fitPreprocessor(data.rows, setup.features);
    const X = E.transform(pp, data.rows); const K = data.classes ? data.classes.length : 1;
    const M = MODELS[algo]; const p = Object.assign({}, M.defaults, params || {});
    const model = M.fit(X, data.y, setup.task, K, p, rng(setup.seed || 7), onProgress ? f => onProgress({ stage: 'final', frac: f }) : null);
    // reference profile for drift checks
    const ref = {};
    for (const f of setup.features) {
      if (f.type === 'numeric') { const v = data.rows.map(r => parseNum(r[f.name])).filter(x => !isNaN(x)).sort((a, b) => a - b); const edges = Array.from(new Set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(q => quantile(v, q)))); const props = numBins(v, edges).map(c => c / Math.max(v.length, 1)); ref[f.name] = { type: 'numeric', edges, props, mean: mean(v), missing: 1 - v.length / data.rows.length }; }
      else if (f.type === 'categorical') { const c = {}; data.rows.forEach(r => { const v = isMissing(r[f.name]) ? '(missing)' : String(r[f.name]).trim(); c[v] = (c[v] || 0) + 1; }); for (const k in c) c[k] /= data.rows.length; ref[f.name] = { type: 'categorical', freq: c }; }
    }
    let targetRef;
    if (setup.task === 'regression') targetRef = { mean: mean(data.y), std: std(data.y) };
    else targetRef = { freq: data.classes.map((_, k) => data.y.filter(v => v === k).length / data.y.length) };
    return { algo, params: p, model, preprocessor: pp, classes: data.classes, nRows: data.rows.length, reference: ref, targetRef };
  };

  E.predict = function (saved, rows) {
    const M = MODELS[saved.algo];
    const X = E.transform(saved.preprocessor, rows);
    const out = M.predict(saved.model, X, saved.task);
    if (saved.task === 'regression') return out.map(v => ({ value: v }));
    return out.map(p => {
      let k = argmax(p);
      if (saved.classes.length === 2 && saved.threshold !== undefined && saved.threshold !== null) k = p[1] >= saved.threshold ? 1 : 0;
      return { value: saved.classes[k], probs: p };
    });
  };

  // contribution of each feature for one row: change in prediction when the feature is reset to a typical value
  E.explainRow = function (saved, row, reference) {
    const baseOut = E.predict(saved, [row])[0];
    const val = o => saved.task === 'regression' ? o.value : o.probs[saved.classes.indexOf(baseOut.value)];
    const base = val(baseOut); const out = [];
    for (const s of saved.preprocessor.steps) {
      const r2 = Object.assign({}, row);
      r2[s.name] = s.type === 'numeric' ? s.impute : s.type === 'categorical' ? s.mode : r2[s.name];
      if (s.type === 'date') continue;
      const v = val(E.predict(saved, [r2])[0]);
      out.push({ feature: s.name, effect: base - v });
    }
    return out.sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
  };

  // ---------- evaluation of a saved model on new labeled data + drift ----------
  E.checkModel = function (saved, table) {
    const report = { drift: [], rows: table.rows.length };
    for (const f in saved.reference) {
      const ref = saved.reference[f]; const vals = table.rows.map(r => r[f]);
      if (!table.columns.includes(f)) { report.drift.push({ feature: f, missingColumn: true, psi: NaN, level: 'critical' }); continue; }
      let psi = 0;
      if (ref.type === 'numeric') {
        const xs = vals.map(parseNum).filter(x => !isNaN(x)); const bins = numBins(xs, ref.edges);
        for (let b = 0; b < bins.length; b++) { const a = Math.max(bins[b] / Math.max(xs.length, 1), 1e-4), e = Math.max(ref.props[b], 1e-4); psi += (a - e) * Math.log(a / e); }
        const nm = mean(vals.map(parseNum).filter(x => !isNaN(x)));
        report.drift.push({ feature: f, psi, refMean: ref.mean, newMean: nm, level: psiLevel(psi) });
      } else {
        const c = {}; vals.forEach(v => { const k = isMissing(v) ? '(missing)' : String(v).trim(); c[k] = (c[k] || 0) + 1; });
        const keys = new Set(Object.keys(c).concat(Object.keys(ref.freq)));
        keys.forEach(k => { const a = Math.max((c[k] || 0) / vals.length, 1e-4), e = Math.max(ref.freq[k] || 0, 1e-4); psi += (a - e) * Math.log(a / e); });
        const unseen = Object.keys(c).filter(k => !(k in ref.freq));
        report.drift.push({ feature: f, psi, unseen: unseen.slice(0, 5), level: psiLevel(psi) });
      }
    }
    report.drift.sort((a, b) => (b.psi || 99) - (a.psi || 99));
    if (table.columns.includes(saved.target) && saved.task !== 'forecast') {
      const labeled = table.rows.filter(r => !isMissing(r[saved.target]));
      if (labeled.length >= 10) {
        const preds = E.predict(saved, labeled);
        if (saved.task === 'regression') report.performance = E.regressionMetrics(labeled.map(r => parseNum(r[saved.target])), preds.map(p => p.value));
        else {
          const ok = labeled.map((r, i) => ({ y: saved.classes.indexOf(String(r[saved.target]).trim()), p: preds[i].probs })).filter(o => o.y >= 0);
          if (ok.length) report.performance = stripConf(E.classificationMetrics(ok.map(o => o.y), ok.map(o => o.p), saved.classes.length, saved.threshold));
        }
        report.labeledRows = labeled.length;
      }
    }
    return report;
  };
  function numBins(xs, edges) { const bins = new Array(edges.length + 1).fill(0); for (const x of xs) { let b = 0; while (b < edges.length && x > edges[b]) b++; bins[b]++; } return bins; }
  function psiLevel(p) { return p > 0.25 ? 'serious' : p > 0.1 ? 'warning' : 'good'; }

  // ---------- time series ----------
  const DAY = 86400000;
  function inferFreq(times) {
    const d = []; for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
    d.sort((a, b) => a - b); const med = d[Math.floor(d.length / 2)] || DAY;
    if (med < DAY * 0.9) return { unit: 'hour', step: 3600000, season: 24, label: 'hourly' };
    if (med < DAY * 1.5) return { unit: 'day', step: DAY, season: 7, label: 'daily' };
    if (med < DAY * 8) return { unit: 'week', step: 7 * DAY, season: 52, label: 'weekly' };
    if (med < DAY * 35) return { unit: 'month', step: 0, season: 12, label: 'monthly' };
    if (med < DAY * 100) return { unit: 'quarter', step: 0, season: 4, label: 'quarterly' };
    return { unit: 'year', step: 0, season: 1, label: 'yearly' };
  }
  function addPeriod(t, freq, k) {
    if (freq.step) return t + freq.step * k;
    const d = new Date(t); const months = freq.unit === 'month' ? k : freq.unit === 'quarter' ? 3 * k : 12 * k;
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + months, d.getUTCDate());
  }
  function periodKey(t, freq) {
    const d = new Date(t);
    if (freq.unit === 'hour') return Math.floor(t / 3600000) * 3600000;
    if (freq.unit === 'day') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    if (freq.unit === 'week') return t; // weekly data keeps its own anchor
    if (freq.unit === 'month') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
    if (freq.unit === 'quarter') return Date.UTC(d.getUTCFullYear(), Math.floor(d.getUTCMonth() / 3) * 3, 1);
    return Date.UTC(d.getUTCFullYear(), 0, 1);
  }
  E.buildSeries = function (table, dateCol, target, agg) {
    const pts = [];
    for (const r of table.rows) { const t = parseDate(r[dateCol]); const v = parseNum(r[target]); if (!isNaN(t) && !isNaN(v)) pts.push([t, v]); }
    if (pts.length < 10) throw new Error('Fewer than 10 rows have both a valid date and a numeric ' + target + '.');
    pts.sort((a, b) => a[0] - b[0]);
    const freq = inferFreq(Array.from(new Set(pts.map(p => p[0]))));
    const m = new Map();
    for (const [t, v] of pts) { const k = periodKey(t, freq); const e = m.get(k) || { s: 0, c: 0 }; e.s += v; e.c++; m.set(k, e); }
    const keys = Array.from(m.keys()).sort((a, b) => a - b);
    const times = [], values = []; let filled = 0;
    let t = keys[0]; let guard = 0;
    const lookup = new Map(keys.map(k => [k, m.get(k)]));
    if (freq.unit === 'week') { keys.forEach(k => { const e = lookup.get(k); times.push(k); values.push(agg === 'mean' ? e.s / e.c : e.s); }); }
    else {
      while (t <= keys[keys.length - 1] && guard++ < 200000) {
        const e = lookup.get(t);
        times.push(t); values.push(e ? (agg === 'mean' ? e.s / e.c : e.s) : NaN); if (!e) filled++;
        t = addPeriod(t, freq, 1);
      }
    }
    for (let i = 0; i < values.length; i++) if (isNaN(values[i])) { let a = i - 1, b = i + 1; while (b < values.length && isNaN(values[b])) b++; const va = a >= 0 ? values[a] : values[b], vb = b < values.length ? values[b] : va; values[i] = va + (vb - va) / (b - a); }
    const duplicates = pts.length - keys.length;
    return { times, values, freq, filled, duplicates };
  };
  function tsDefaultLags(freq) { return { hour: [1, 2, 3, 24, 48, 168], day: [1, 2, 3, 7, 14, 28], week: [1, 2, 4, 52], month: [1, 2, 3, 12], quarter: [1, 2, 4], year: [1, 2] }[freq.unit]; }
  function tsFeatures(hist, t, freq, lags, tIndex) {
    const n = hist.length; const f = [];
    for (const L of lags) f.push(n - L >= 0 ? hist[n - L] : hist[0]);
    const w = Math.min(freq.season > 1 ? freq.season : 3, n); let s = 0; for (let i = n - w; i < n; i++) s += hist[i]; f.push(s / w);
    const d = new Date(t);
    if (freq.unit === 'hour') { const h = d.getUTCHours(); f.push(Math.sin(2 * Math.PI * h / 24), Math.cos(2 * Math.PI * h / 24), d.getUTCDay() === 0 || d.getUTCDay() === 6 ? 1 : 0); }
    if (freq.unit === 'day') { for (let k = 0; k < 7; k++) f.push(d.getUTCDay() === k ? 1 : 0); }
    if (freq.unit === 'day' || freq.unit === 'week' || freq.unit === 'month') { const doy = (t - Date.UTC(d.getUTCFullYear(), 0, 1)) / DAY; f.push(Math.sin(2 * Math.PI * doy / 365.25), Math.cos(2 * Math.PI * doy / 365.25)); }
    if (freq.unit === 'month') for (let k = 0; k < 12; k++) f.push(d.getUTCMonth() === k ? 1 : 0);
    if (freq.unit === 'quarter') for (let k = 0; k < 4; k++) f.push(Math.floor(d.getUTCMonth() / 3) === k ? 1 : 0);
    f.push(tIndex);
    return f;
  }
  function fitTrend(values) { const n = values.length; const xs = range(n); const xm = (n - 1) / 2, ym = mean(values); let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (i - xm) * (values[i] - ym); sxx += (i - xm) * (i - xm); } const b = sxx ? sxy / sxx : 0; return { a: ym - b * xm, b }; }
  // train a forecaster on values[0..end) and return it
  function fitForecaster(series, end, algo, params, opts, r) {
    const { times, freq } = series; const lags = opts.lags || tsDefaultLags(freq);
    const vals = series.values.slice(0, end);
    const detrend = opts.detrend !== false && algo !== 'linear';
    const trend = detrend ? fitTrend(vals) : { a: 0, b: 0 };
    const res = vals.map((v, i) => v - (trend.a + trend.b * i));
    const start = Math.max(Math.max.apply(null, lags), 3);
    const X = [], y = [];
    for (let i = start; i < end; i++) { X.push(tsFeatures(res.slice(0, i), times[i], freq, lags, detrend ? 0 : i / end)); y.push(res[i]); }
    // scale
    const d = X[0].length; const mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    X.forEach(x => x.forEach((v, j) => mu[j] += v / X.length)); X.forEach(x => x.forEach((v, j) => sd[j] += (v - mu[j]) * (v - mu[j]) / X.length));
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    const Xs = X.map(x => Float64Array.from(x.map((v, j) => (v - mu[j]) / sd[j])));
    const M = MODELS[algo]; const p = Object.assign({}, M.defaults, params || {});
    const model = M.fit(Xs, y, 'regression', 1, p, r);
    const resid = M.predict(model, Xs, 'regression').map((v, i) => y[i] - v);
    return { algo, params: p, model, lags, trend, detrend, mu, sd, end, residStd: std(resid), trainN: end };
  }
  function forecastFrom(fc, series, hist, lastTime, h) {
    const M = MODELS[fc.algo]; const { freq } = series; const out = [];
    const res = hist.map((v, i) => v - (fc.trend.a + fc.trend.b * i));
    let t = lastTime; const n0 = hist.length;
    for (let s = 0; s < h; s++) {
      t = addPeriod(t, freq, 1);
      const x = tsFeatures(res, t, freq, fc.lags, fc.detrend ? 0 : (n0 + s) / fc.end).map((v, j) => (v - fc.mu[j]) / fc.sd[j]);
      const pr = M.predict(fc.model, [Float64Array.from(x)], 'regression')[0];
      res.push(pr); const idx = n0 + s;
      out.push({ t, value: pr + fc.trend.a + fc.trend.b * idx });
    }
    return out;
  }
  function seasonalNaive(series, end, h) { const s = series.freq.season > 1 && end > series.freq.season ? series.freq.season : 1; const v = series.values; const out = []; for (let i = 0; i < h; i++) out.push(v[end - s + (i % s)]); return out; }

  // setup: {dateCol, target, agg, horizon, algos, params, seed, detrend}
  E.runForecast = function (table, setup, onProgress) {
    const prog = onProgress || function () {};
    const series = E.buildSeries(table, setup.dateCol, setup.target, setup.agg || 'sum');
    const n = series.values.length; const h = Math.max(1, Math.min(setup.horizon || series.freq.season || 12, Math.floor(n / 4)));
    if (n < 24) throw new Error('The series has only ' + n + ' periods. Forecasting needs at least 24.');
    const end = n - h; const actual = series.values.slice(end);
    const algos = (setup.algos && setup.algos.length ? setup.algos : ['baseline', 'linear', 'boosting']).filter(a => a !== 'knn' && a !== 'tree' || true);
    const results = [];
    algos.forEach((a, ai) => {
      let pred;
      const t0 = Date.now();
      if (a === 'baseline') pred = seasonalNaive(series, end, h);
      else { const fc = fitForecaster(series, end, a, (setup.params || {})[a], setup, rng((setup.seed || 7) + ai)); pred = forecastFrom(fc, series, series.values.slice(0, end), series.times[end - 1], h).map(p => p.value); }
      const m = E.regressionMetrics(actual, pred);
      results.push({ key: a, algo: a, label: a === 'baseline' ? 'Seasonal naive' : MODELS[a].label, valid: m, pred, ms: Date.now() - t0, score: m[setup.metric || 'mae'], params: a === 'baseline' ? {} : Object.assign({}, MODELS[a].defaults, (setup.params || {})[a] || {}) });
      prog({ stage: 'train', algo: a, i: ai + 1, of: algos.length });
    });
    const metric = setup.metric || 'mae';
    const sorted = results.slice().sort((a, b) => better(metric, a.score, b.score) ? -1 : 1);
    const best = sorted[0];
    const notes = [];
    const base = results.find(r => r.algo === 'baseline');
    if (base && best.algo === 'baseline') notes.push({ level: 'warning', title: 'Seasonal naive won', text: 'Repeating last season beat the learned models. The series may be dominated by its seasonal pattern; try a longer history or a different horizon.' });
    if (series.filled > 0) notes.push({ level: 'info', title: series.filled + ' missing period' + (series.filled > 1 ? 's' : '') + ' filled', text: 'Gaps in the dates were filled by straight-line interpolation.' });
    if (series.duplicates > 0) notes.push({ level: 'info', title: 'Rows combined per ' + series.freq.unit, text: series.duplicates + ' rows shared a period with another row and were combined using ' + (setup.agg || 'sum') + '.' });
    return { task: 'forecast', metric, series: { times: series.times, values: series.values, freq: series.freq }, horizon: h, holdoutStart: end, results: results.map(r => ({ key: r.key, algo: r.algo, label: r.label, valid: r.valid, pred: r.pred, ms: r.ms, score: r.score, params: r.params })), bestKey: best.key, bestAlgo: best.algo, bestParams: best.params, diagnostics: notes };
  };
  // holdout score for one algorithm without touching the in-memory experiment (used by retraining)
  E.holdoutScore = function (table, setup, algo, params) {
    const data = E.prepareData(table, setup); const task = setup.task; const K = data.classes ? data.classes.length : 1;
    const r = rng(setup.seed || 7);
    const orderBy = setup.orderBy ? data.rows.map(row => { const d = parseDate(row[setup.orderBy]); return isNaN(d) ? parseNum(row[setup.orderBy]) : d; }) : null;
    const sp = splitIndices(data.rows.length, data.y, task, setup.validFrac || 0.2, r, orderBy);
    const pp = E.fitPreprocessor(sp.train.map(i => data.rows[i]), setup.features);
    const Xtr = E.transform(pp, sp.train.map(i => data.rows[i])), Xva = E.transform(pp, sp.valid.map(i => data.rows[i]));
    const M = MODELS[algo]; const model = M.fit(Xtr, sp.train.map(i => data.y[i]), task, K, Object.assign({}, M.defaults, params || {}), rng(setup.seed || 7));
    const P = M.predict(model, Xva, task); const yv = sp.valid.map(i => data.y[i]);
    return { valid: stripConf(task === 'regression' ? E.regressionMetrics(yv, P) : E.classificationMetrics(yv, P, K)), nRows: data.rows.length };
  };
  E.trainForecastCandidate = function (table, setup, algo, params) {
    const series = E.buildSeries(table, setup.dateCol, setup.target, setup.agg || 'sum');
    const n = series.values.length; const h = Math.max(1, Math.min(setup.horizon || series.freq.season || 12, Math.floor(n / 4)));
    const end = n - h; const t0 = Date.now();
    const fc = fitForecaster(series, end, algo, params, setup, rng((setup.seed || 7) + 77));
    const pred = forecastFrom(fc, series, series.values.slice(0, end), series.times[end - 1], h).map(p => p.value);
    const m = E.regressionMetrics(series.values.slice(end), pred);
    return { algo, label: MODELS[algo].label + ' \xb7 tuned', valid: m, pred, ms: Date.now() - t0, score: m[setup.metric || 'mae'], params: fc.params, tuned: true };
  };
  E.tuneForecast = function (table, setup, algo, trials, onProgress) {
    const prog = onProgress || function () {};
    const series = E.buildSeries(table, setup.dateCol, setup.target, setup.agg || 'sum');
    const n = series.values.length; const h = Math.max(1, Math.min(setup.horizon || series.freq.season || 12, Math.floor(n / 4)));
    const M = MODELS[algo]; const r = rng((setup.seed || 7) + 500); const metric = setup.metric || 'mae';
    const origins = [n - 3 * h, n - 2 * h, n - h].filter(e => e > Math.max(30, n * 0.4));
    if (!origins.length) origins.push(n - h);
    const cand = [Object.assign({}, M.defaults, (setup.params || {})[algo] || {})];
    for (let t = 1; t < trials; t++) { const p = Object.assign({}, cand[0]); for (const k in M.space) { const s = M.space[k]; p[k] = s.type === 'int' ? s.min + Math.floor(r() * (s.max - s.min + 1)) : s.type === 'log' ? round(Math.exp(Math.log(s.min) + r() * (Math.log(s.max) - Math.log(s.min))), 4) : round(s.min + r() * (s.max - s.min), 3); } cand.push(p); }
    const out = cand.map((p, t) => {
      const scores = origins.map(e => { const fc = fitForecaster(series, e, algo, p, setup, rng(t + 3)); const pr = forecastFrom(fc, series, series.values.slice(0, e), series.times[e - 1], h).map(q => q.value); return E.regressionMetrics(series.values.slice(e, e + h), pr)[metric]; });
      prog({ stage: 'tune', i: t + 1, of: cand.length });
      return { params: p, score: mean(scores), spread: std(scores), isCurrent: t === 0 };
    });
    out.sort((a, b) => better(metric, a.score, b.score) ? -1 : 1);
    return { algo, metric, trials: out, best: out[0].params };
  };
  E.fitForecastFinal = function (table, setup, algo, params) {
    const series = E.buildSeries(table, setup.dateCol, setup.target, setup.agg || 'sum');
    const n = series.values.length;
    let fc = null;
    if (algo !== 'baseline') fc = fitForecaster(series, n, algo, params, setup, rng(setup.seed || 7));
    return { algo, params: fc ? fc.params : {}, fc, series: { times: series.times, values: series.values, freq: series.freq }, residStd: fc ? fc.residStd : std(series.values.slice(1).map((v, i) => v - series.values[i])) };
  };
  E.forecast = function (saved, h, table) {
    let series = saved.series;
    if (table) { const s = E.buildSeries(table, saved.dateCol, saved.target, saved.agg || 'sum'); series = { times: s.times, values: s.values, freq: s.freq }; }
    const n = series.values.length;
    let pts;
    if (saved.algo === 'baseline') { const pv = seasonalNaive(series, n, h); let t = series.times[n - 1]; pts = pv.map(v => { t = addPeriod(t, series.freq, 1); return { t, value: v }; }); }
    else pts = forecastFrom(saved.fc, series, series.values.slice(), series.times[n - 1], h);
    const sdv = saved.residStd || 0;
    pts.forEach((p, i) => { const w = 1.28 * sdv * Math.sqrt(1 + i * 0.35); p.lo = p.value - w; p.hi = p.value + w; });
    return { history: { times: series.times, values: series.values }, points: pts, freq: series.freq };
  };

  // ---------- sample datasets ----------
  E.samples = {
    homes: { title: 'Home sale prices', blurb: '900 home sales. Predict the sale price (a number).', target: 'sale_price', task: 'regression' },
    churn: { title: 'Customer churn', blurb: '1,400 subscribers. Predict who cancels (yes/no).', target: 'churned', task: 'classification' },
    sales: { title: 'Daily store sales', blurb: 'Two years of daily unit sales. Forecast the coming weeks.', target: 'units_sold', task: 'forecast', dateCol: 'date' }
  };
  E.makeSample = function (key) {
    const r = rng(key === 'homes' ? 11 : key === 'churn' ? 23 : 37);
    const rows = [];
    if (key === 'homes') {
      const hoods = [['Riverside', 1.25], ['Old Town', 1.1], ['Northgate', 1.0], ['Meadowbrook', 0.92], ['Eastfield', 0.8]];
      for (let i = 0; i < 900; i++) {
        const h = r.pick(hoods); const beds = 1 + r.int(5); const sqft = Math.round(520 + beds * 380 + r.normal() * 260);
        const baths = Math.max(1, Math.min(4, Math.round(beds * 0.6 + r() * 1.2) / 1)); const year = 1935 + r.int(88);
        const lot = Math.round(Math.max(1200, 3200 + sqft * 1.4 + r.normal() * 1800)); const garage = r() < 0.65 ? 'yes' : 'no';
        const cond = r() < 0.2 ? 'Fair' : r() < 0.75 ? 'Good' : 'Excellent';
        const m = 1 + r.int(12); const yr = 2023 + (r() < 0.5 ? 1 : 0);
        let price = (48000 + sqft * 148 + baths * 9500 + lot * 2.1 + (year - 1935) * 520 + (garage === 'yes' ? 14000 : 0)) * h[1] * (cond === 'Fair' ? 0.88 : cond === 'Excellent' ? 1.09 : 1) * (1 + r.normal() * 0.07);
        rows.push({ listing_id: 100001 + i, neighborhood: h[0], sqft, bedrooms: beds, bathrooms: baths, year_built: year, lot_size: r() < 0.06 ? '' : lot, garage, condition: cond, sold_date: yr + '-' + String(m).padStart(2, '0') + '-' + String(1 + r.int(28)).padStart(2, '0'), sale_price: Math.round(price / 100) * 100 });
      }
      return { columns: ['listing_id', 'neighborhood', 'sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'condition', 'sold_date', 'sale_price'], rows };
    }
    if (key === 'churn') {
      for (let i = 0; i < 1400; i++) {
        const contract = r() < 0.52 ? 'Month-to-month' : r() < 0.55 ? 'One year' : 'Two year';
        const tenure = Math.max(1, Math.round(contract === 'Month-to-month' ? 4 + r() * 30 : 12 + r() * 60));
        const internet = r() < 0.45 ? 'Fiber' : r() < 0.75 ? 'DSL' : 'None';
        const charges = Math.round((internet === 'Fiber' ? 82 : internet === 'DSL' ? 58 : 24) + r.normal() * 11);
        const pay = r.pick(['Card', 'Bank transfer', 'Electronic check', 'Mailed check']);
        const calls = Math.max(0, Math.round(r() * r() * 7)); const age = 19 + r.int(60);
        const z = -1.2 + (contract === 'Month-to-month' ? 1.5 : contract === 'One year' ? -0.3 : -1.6) - 0.035 * tenure + 0.018 * (charges - 60) + 0.42 * calls + (pay === 'Electronic check' ? 0.55 : 0) + (internet === 'Fiber' ? 0.35 : 0) + r.normal() * 0.6;
        rows.push({ customer_id: 'C' + (50000 + i * 7), contract, tenure_months: tenure, internet, monthly_charges: charges, payment_method: pay, support_calls: calls, age, churned: r() < sigmoid(z) ? 'Yes' : 'No' });
      }
      return { columns: ['customer_id', 'contract', 'tenure_months', 'internet', 'monthly_charges', 'payment_method', 'support_calls', 'age', 'churned'], rows };
    }
    const start = Date.UTC(2024, 8, 1);
    for (let i = 0; i < 730; i++) {
      const t = start + i * DAY; const d = new Date(t); const dow = d.getUTCDay();
      const doy = (t - Date.UTC(d.getUTCFullYear(), 0, 1)) / DAY;
      const promo = r() < 0.08 ? 1 : 0;
      const v = 140 + i * 0.09 + [34, -12, -16, -9, 0, 21, 45][dow] + 26 * Math.sin(2 * Math.PI * (doy - 80) / 365.25) + (d.getUTCMonth() === 11 ? 38 : 0) + promo * 40 + r.normal() * 11;
      rows.push({ date: d.toISOString().slice(0, 10), units_sold: Math.max(0, Math.round(v)), promo, store: 'Main St' });
    }
    return { columns: ['date', 'store', 'promo', 'units_sold'], rows };
  };

  // ---------- worker protocol ----------
  const PROGRESS_FNS = { runExperiment: 1, tune: 1, fitFinal: 1, runForecast: 1, tuneForecast: 1, trainCandidate: 1 };
  const isWorker =typeof self !== 'undefined' && typeof importScripts === 'function' && typeof window === 'undefined';
  if (isWorker) {
    self.onmessage = function (ev) {
      const { id, fn, args } = ev.data;
      try {
        const a = (args || []).slice();
        if (PROGRESS_FNS[fn]) { while (a.length < E[fn].length - 1) a.push(undefined); a.push(p => self.postMessage({ id, progress: p })); }
        const res = E[fn].apply(null, a);
        self.postMessage({ id, result: res });
      } catch (err) { self.postMessage({ id, error: String(err && err.message || err) }); }
    };
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = E;
  else root.TabulaEngine = E;
})(typeof self !== 'undefined' ? self : this);
