// Engine smoke tests: run with `npm test` (Node 18+, no dependencies).
const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../engine.js');

function features(profile, target) {
  return profile.columns.filter(c => c.role === 'feature' && c.name !== target).map(c => ({ name: c.name, type: c.type }));
}

test('CSV round trip and profiling', () => {
  const t = E.makeSample('homes');
  const back = E.parseCSV(E.toCSV(t.columns, t.rows));
  assert.equal(back.rows.length, t.rows.length);
  const prof = E.profile(back);
  const col = n => prof.columns.find(c => c.name === n);
  assert.equal(col('listing_id').type, 'id');
  assert.equal(col('listing_id').role, 'ignore');
  assert.equal(col('sqft').type, 'numeric');
  assert.equal(col('sold_date').type, 'date');
  assert.equal(col('neighborhood').type, 'categorical');
  assert.equal(E.suggestTask(prof, 'sale_price').task, 'regression');
});

test('value parsing', () => {
  assert.equal(E.parseNum('$1,200'), 1200);
  assert.equal(E.parseNum('(50)'), -50);
  assert.ok(isNaN(E.parseNum('N/A')));
  assert.equal(new Date(E.parseDate('2024-03-05')).toISOString().slice(0, 10), '2024-03-05');
});

test('regression beats the baseline', () => {
  const t = E.makeSample('homes'); const prof = E.profile(t);
  const res = E.runExperiment(t, { target: 'sale_price', task: 'regression', features: features(prof, 'sale_price'), algos: ['baseline', 'linear', 'boosting'] });
  const r = k => res.results.find(x => x.algo === k).valid;
  assert.ok(r('boosting').r2 > 0.85, 'boosting R2 ' + r('boosting').r2);
  assert.ok(r('linear').rmse < r('baseline').rmse);
  assert.equal(res.inspect.importance[0].feature, 'sqft');
});

test('classification, tuning, saving and prediction', () => {
  const t = E.makeSample('churn'); const prof = E.profile(t);
  const setup = { target: 'churned', task: 'classification', features: features(prof, 'churned'), algos: ['baseline', 'linear', 'forest'] };
  const res = E.runExperiment(t, setup);
  assert.deepEqual(res.classes, ['No', 'Yes']);
  assert.ok(res.results.find(x => x.algo === 'linear').valid.auc > 0.8);
  const tuned = E.tune(t, setup, 'forest', 3);
  assert.equal(tuned.trials.length, 3);
  const saved = JSON.parse(JSON.stringify(Object.assign({ task: 'classification', target: 'churned' }, E.fitFinal(t, setup, 'linear', {}))));
  const preds = E.predict(saved, t.rows.slice(0, 5));
  preds.forEach(p => { assert.ok(['No', 'Yes'].includes(p.value)); assert.ok(Math.abs(p.probs[0] + p.probs[1] - 1) < 1e-9); });
  const drifted = { columns: t.columns, rows: t.rows.slice(0, 300).map(r => Object.assign({}, r, { monthly_charges: +r.monthly_charges + 25 })) };
  const chk = E.checkModel(saved, drifted);
  assert.equal(chk.drift[0].feature, 'monthly_charges');
  assert.equal(chk.drift[0].level, 'serious');
});

test('multiclass classification', () => {
  const r = E.rng(3); const rows = [];
  for (let i = 0; i < 600; i++) { const a = r() * 10, b = r() * 10; rows.push({ a, b, cls: a + b < 7 ? 'low' : a + b < 13 ? 'mid' : 'high' }); }
  const tbl = { columns: ['a', 'b', 'cls'], rows };
  const res = E.runExperiment(tbl, { target: 'cls', task: 'classification', features: [{ name: 'a', type: 'numeric' }, { name: 'b', type: 'numeric' }], algos: ['baseline', 'boosting'] });
  assert.ok(res.results.find(x => x.algo === 'boosting').valid.accuracy > 0.9);
});

test('forecasting beats seasonal naive and extends the series', () => {
  const t = E.makeSample('sales');
  const setup = { dateCol: 'date', target: 'units_sold', horizon: 28, algos: ['baseline', 'linear'] };
  const res = E.runForecast(t, setup);
  assert.equal(res.series.freq.unit, 'day');
  const mae = k => res.results.find(x => x.algo === k).valid.mae;
  assert.ok(mae('linear') < mae('baseline'));
  const saved = Object.assign({ task: 'forecast', target: 'units_sold', dateCol: 'date' }, E.fitForecastFinal(t, setup, 'linear', {}));
  const out = E.forecast(JSON.parse(JSON.stringify(saved)), 14);
  assert.equal(out.points.length, 14);
  assert.ok(out.points[0].t > out.history.times[out.history.times.length - 1]);
  assert.ok(out.points.every(p => p.lo <= p.value && p.value <= p.hi));
});

test('row objects from JSON or Parquet are normalized', () => {
  const t = E.fromObjects([
    { id: 1n, when: new Date(Date.UTC(2024, 4, 2)), at: new Date(Date.UTC(2024, 4, 2, 13, 5)), ok: true, tags: ['a'], v: null },
    { id: 2n, extra: 'x' },
  ]);
  assert.deepEqual(t.columns, ['id', 'when', 'at', 'ok', 'tags', 'v', 'extra']);
  assert.deepEqual(t.rows[0], { id: 1, when: '2024-05-02', at: '2024-05-02 13:05:00', ok: 'true', tags: '["a"]', v: '', extra: '' });
  assert.equal(t.rows[1].when, '');
  const prof = E.profile(t);
  assert.equal(prof.columns.find(c => c.name === 'when').type, 'date');
});
