'use strict';

const byId = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? value.toLocaleString('zh-CN') : '—';
const decimal = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const percent = value => Number.isFinite(value) ? (100 * value).toFixed(1) + '%' : '—';
const splitName = split => split === 'test' ? '测试集' : '验证集';
const padIndex = value => String(value).padStart(3, '0');
let report, group, selected, visible = [], importVersion = 0;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showNotice(message) {
  byId('notice').textContent = message;
  byId('notice').hidden = !message;
}

function words(text) {
  const trimmed = text.trim().toLowerCase();
  return trimmed ? trimmed.split(/\s+/u) : [];
}

function overlap(a, b) {
  let previous = new Uint32Array(b.length + 1);
  let current = new Uint32Array(b.length + 1);
  for (const token of a) {
    for (let j = 1; j <= b.length; j++) {
      current[j] = token === b[j - 1] ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
    }
    [previous, current] = [current, previous];
  }
  return a.length + b.length ? 2 * previous[b.length] / (a.length + b.length) : 0;
}

// Match the evaluator's Unicode word boundary and compare numeric multisets, including duplicates.
function numericStrings(text) {
  return (text.toLowerCase().match(/(?<![\p{L}\p{N}_])[+-]?\p{Nd}+(?:[.,]\p{Nd}+)*(?:%|e[+-]?\p{Nd}+)?/gu) || []).sort();
}

function analyzeGroup(raw) {
  const m = raw.metrics, rows = raw.rows;
  if (!m || !['test', 'validation'].includes(m.split) || m.rewrite_depth !== 'final' || !Array.isArray(rows) || !rows.length || m.n !== rows.length) {
    throw new Error('需要成对的最终改写预测与指标，且样本数须一致。');
  }
  const ids = new Set(), frequencies = new Map();
  for (const row of rows) {
    if (!['id','source','reference','prediction'].every(key => typeof row[key] === 'string') || !row.id || ids.has(row.id)) {
      throw new Error('逐条评估含缺失字段或重复的样本 ID。');
    }
    ids.add(row.id);
    const key = row.prediction.trim();
    frequencies.set(key, (frequencies.get(key) || 0) + 1);
  }
  const samples = rows.map((row, i) => {
    const target = words(row.reference), inputNumbers = numericStrings(row.source), outputNumbers = numericStrings(row.prediction);
    return {...row, index: i + 1, rouge: overlap(words(row.prediction), target), baseline: overlap(words(row.source), target),
      hasNumbers: inputNumbers.length > 0, numbersMatch: JSON.stringify(inputNumbers) === JSON.stringify(outputNumbers),
      repeats: frequencies.get(row.prediction.trim()), searchable: [row.id, row.source, row.reference, row.prediction].join('\n').toLowerCase()};
  });
  return {name: raw.name, metrics: m, samples, below: samples.filter(r => r.rouge < r.baseline).length,
    zeros: samples.filter(r => r.rouge === 0).length, numeric: samples.filter(r => r.hasNumbers).length,
    numericKept: samples.filter(r => r.hasNumbers && r.numbersMatch).length, unique: frequencies.size};
}

const filters = {all: () => true, below: r => r.rouge < r.baseline, zero: r => r.rouge === 0,
  numbers: r => !r.numbersMatch, repeated: r => r.repeats > 1, limit: r => r.generation_limit_reached === true};

function tableRow(cells) {
  const tr = el('tr');
  for (const cell of cells) tr.append(el('td', cell));
  return tr;
}

function kpis(id, items) {
  byId(id).replaceChildren(...items.map(([label, value, foot]) => {
    const card = el('div', undefined, 'kpi');
    card.append(el('span', label, 'klabel'), el('strong', value, 'kvalue'), el('span', foot, 'kfoot'));
    return card;
  }));
}

function groupMetrics() {
  const m = group.metrics, base = m.copy_baseline || {}, n = group.samples.length;
  kpis('kpis', [
    ['模型 ROUGE-L', decimal(m.whitespace_rouge_l_f1), '复制原句 ' + decimal(base.whitespace_rouge_l_f1)],
    ['低于复制基线', group.below + ' / ' + n, '逐条对照人工参考'],
    ['不同预测文本', number(group.unique), n + ' 条原始生成结果'],
    ['数字完整保留', group.numericKept + ' / ' + group.numeric, '仅统计原句含数字的样本']
  ]);
  byId('split-finding').textContent = '本组共 ' + n + ' 条。模型 ROUGE-L 低于复制基线的有 ' + group.below + ' 条，模型得分为零的有 ' + group.zeros + ' 条。';
  const metrics = [
    ['ROUGE-L（0–1）', decimal(m.whitespace_rouge_l_f1), decimal(base.whitespace_rouge_l_f1)],
    ['SacreBLEU（0–100）', decimal(m.sacrebleu, 3), decimal(base.sacrebleu, 3)],
    ['原样复制率', percent(m.copy_rate), percent(base.copy_rate)],
    ['数字保留率（全组）', percent(m.number_multiset_preservation_rate), percent(base.number_multiset_preservation_rate)],
    ['参考 token 交叉熵', decimal(m.cross_entropy), '—'],
    ['触及生成上限（条）', number(m.generation_limit_reached_count), '—'],
    ['输入 / 参考被截断（条）', number(m.input_truncated_count) + ' / ' + number(m.target_truncated_count), '—']
  ];
  byId('metrics-table').replaceChildren(...metrics.map(tableRow));
}

// Keep original whitespace and punctuation for display; align only the non-whitespace tokens.
function tokenized(text) {
  const pieces = text.match(/\s+|[\p{L}\p{M}\p{N}_]+|[^\s\p{L}\p{M}\p{N}_]/gu) || [];
  const tokens = pieces.filter(piece => !/^\s/u.test(piece));
  return {pieces, tokens};
}

function matchedTokens(a, b) {
  const width = b.length + 1, lcs = new Uint32Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      const pos = i * width + j;
      lcs[pos] = a[i] === b[j] ? 1 + lcs[pos + width + 1] : Math.max(lcs[pos + width], lcs[pos + 1]);
    }
  }
  const left = new Set(), right = new Set();
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    const pos = i * width + j;
    if (a[i] === b[j]) { left.add(i++); right.add(j++); }
    else if (lcs[pos + width] >= lcs[pos + 1]) i++;
    else j++;
  }
  return [left, right];
}

function highlight(id, pieces, matched, className) {
  if (!pieces.length) return;
  const fragment = document.createDocumentFragment();
  let tokenIndex = 0;
  for (const piece of pieces) {
    const changed = !/^\s/u.test(piece) && !matched.has(tokenIndex++);
    fragment.append(changed ? el('span', piece, className) : document.createTextNode(piece));
  }
  byId(id).replaceChildren(fragment);
}

function showTexts() {
  for (const key of ['source','reference','prediction']) {
    byId(key).textContent = selected[key] || '（空文本）';
    byId(key).classList.toggle('blank', !selected[key]);
  }
  const mode = byId('diff').value;
  byId('diff-key').hidden = mode === 'off';
  if (mode === 'off') return;
  const [a,b] = mode.split('-');
  const left = tokenized(selected[a]), right = tokenized(selected[b]);
  const matches = matchedTokens(left.tokens, right.tokens);
  highlight(a, left.pieces, matches[0], 'removed');
  highlight(b, right.pieces, matches[1], 'added');
}

function itemStat(label, value) {
  const node = el('span');
  node.append(document.createTextNode(label + ' '), el('b', value));
  return node;
}

function showSample(scroll = false) {
  const hasRow = Boolean(selected);
  byId('not-found').hidden = hasRow;
  byId('sample-detail').hidden = !hasRow;
  if (!hasRow) return;
  const active = byId('samples').querySelector('[aria-current]');
  if (active) active.removeAttribute('aria-current');
  const button = byId('samples').querySelector('[data-index="' + selected.index + '"]');
  if (button) {
    button.setAttribute('aria-current', 'true');
    const list = byId('samples'), rect = button.getBoundingClientRect(), bounds = list.getBoundingClientRect();
    if (scroll && rect.top < bounds.top) list.scrollTop -= bounds.top - rect.top;
    if (scroll && rect.bottom > bounds.bottom) list.scrollTop += rect.bottom - bounds.bottom;
  }
  const pos = visible.indexOf(selected);
  byId('prev').disabled = pos === 0;
  byId('next').disabled = pos === visible.length - 1;
  byId('sample-title').textContent = '样本 ' + padIndex(selected.index);
  byId('sample-id').textContent = selected.id;
  showTexts();
  const stats = [itemStat('模型 ROUGE-L', decimal(selected.rouge)), itemStat('复制基线', decimal(selected.baseline)),
    itemStat('数字', selected.numbersMatch ? (selected.hasNumbers ? '保留' : '双方均无') : '不一致'),
    itemStat('生成 token', number(selected.generated_tokens))];
  if (selected.repeats > 1) stats.push(itemStat('同组相同预测', selected.repeats + ' 条'));
  if (selected.generation_limit_reached) stats.push(itemStat('生成', '触及上限'));
  if (selected.input_truncated || selected.target_truncated) stats.push(itemStat('截断', [selected.input_truncated && '原句', selected.target_truncated && '参考'].filter(Boolean).join(' / ')));
  byId('row-metrics').replaceChildren(...stats);
}

function filterSamples() {
  if (!group) return;
  const q = byId('search').value.trim().toLowerCase(), test = filters[byId('filter').value];
  visible = group.samples.filter(row => test(row) && row.searchable.includes(q));
  if (!visible.includes(selected)) selected = visible[0];
  byId('count').textContent = visible.length + ' / ' + group.samples.length;
  byId('samples').replaceChildren(...visible.map(row => {
    const item = el('button', undefined, 'sample-button');
    item.type = 'button'; item.dataset.index = row.index;
    item.append(el('span', '#' + padIndex(row.index), 'idx'), el('span', row.source, 'snippet'));
    item.addEventListener('click', () => {
      selected = row; showSample();
      if (narrowScreen.matches) byId('index-toggle').open = false;
    });
    return item;
  }));
  showSample(true);
}

function selectGroup(nextGroup) {
  group = nextGroup;
  selected = undefined;
  byId('search').value = ''; byId('filter').value = 'all';
  for (const button of byId('split-switch').children) {
    button.setAttribute('aria-pressed', String(button.dataset.name === group.name));
  }
  groupMetrics(); filterSamples();
}

function svgElement(tag, attrs, label) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function chart(id, log, key, color, label) {
  const holder = byId(id), points = log.filter(p => Number.isFinite(p.step) && Number.isFinite(p[key]));
  holder.replaceChildren();
  if (!points.length) { holder.append(el('p','此目录未记录该曲线。','muted')); return; }
  const start = points[0].step, end = points[points.length - 1].step;
  let low = Infinity, high = -Infinity;
  for (const point of points) { low = Math.min(low, point[key]); high = Math.max(high, point[key]); }
  const unit = high - low < 2 ? .5 : 1;
  low = Math.floor(low / unit) * unit; high = Math.ceil(high / unit) * unit;
  if (low === high) high += unit;
  const x = step => 44 + (step - start) / (end - start || 1) * 532;
  const y = value => 178 - (value - low) / (high - low) * 155;
  const svg = svgElement('svg', {viewBox: '0 0 600 210', tabindex: '0', role: 'img', 'aria-label': label + '。左右方向键查看记录。'});
  const line = (x1,y1,x2,y2,stroke) => svgElement('line',{x1,y1,x2,y2,stroke});
  for (let tick = 0; tick <= 3; tick++) {
    const value = low + (high - low) * tick / 3;
    svg.append(line(44,y(value),576,y(value),'#dbe4de'), svgElement('text',{x:34,y:y(value)+4,'text-anchor':'end'},decimal(value,1)));
  }
  for (const step of new Set([start, Math.round((start + end) / 2), end])) {
    svg.append(svgElement('text',{x:x(step),y:202,'text-anchor':'middle'},number(step)));
  }
  const coords = points.map(p => x(p.step).toFixed(2) + ',' + y(p[key]).toFixed(2)).join(' ');
  svg.append(svgElement('polygon',{points:'44,178 ' + coords + ' ' + x(end) + ',178',fill:color,'fill-opacity':'.075'}));
  svg.append(svgElement('polyline',{points:coords,fill:'none',stroke:color,'stroke-width':2.4,'stroke-linejoin':'round'}));
  if (points.length < 50) for (const p of points) svg.append(svgElement('circle',{cx:x(p.step),cy:y(p[key]),r:3,fill:color}));
  const cursor = line(0,23,0,178,'#7b8d84'); cursor.setAttribute('stroke-dasharray','3 4');
  const dot = svgElement('circle',{r:4.5,fill:color,stroke:'white','stroke-width':2});
  const readout = el('p',undefined,'chart-value');
  svg.append(cursor,dot); holder.append(svg,readout);
  let index = 0;
  function show(i) {
    index = Math.max(0, Math.min(i, points.length - 1));
    const p = points[index];
    cursor.setAttribute('x1',x(p.step)); cursor.setAttribute('x2',x(p.step));
    dot.setAttribute('cx',x(p.step)); dot.setAttribute('cy',y(p[key]));
    readout.textContent = '第 ' + number(p.step) + ' 步 · ' + label + ' ' + decimal(p[key]);
  }
  svg.addEventListener('pointermove', event => {
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const step = start + (((event.clientX - rect.left) / rect.width * 600 - 44) / 532) * (end - start);
    let l = 0, r = points.length - 1;
    while (l < r) { const mid = Math.floor((l + r) / 2); if (points[mid].step < step) l = mid + 1; else r = mid; }
    show(l > 0 && step - points[l - 1].step < points[l].step - step ? l - 1 : l);
  });
  svg.addEventListener('keydown', event => {
    const offsets = {ArrowRight:index+1, ArrowLeft:index-1, Home:0, End:points.length-1};
    if (Object.hasOwn(offsets,event.key)) { event.preventDefault(); show(offsets[event.key]); }
  });
  show(points.length - 1);
}

function drawTraining() {
  const r = report.result, val = report.validation;
  kpis('training-kpis',[
    ['已完成更新',number(r.steps),'计划 ' + number(r.requested_steps) + ' 步'],
    ['保存的检查点',number(r.selected_step),'更新步'],
    ['句级训练样本',number(r.train_pairs),'训练句对'],
    ['词级训练样本',number(r.word_training_pairs),'单词监督样本']
  ]);
  chart('train-chart',report.training,'train_loss_ma20','#197565','训练损失 MA20');
  chart('val-chart',val,'validation_cross_entropy','#795a93','P4 验证 CE');
  const insights = [];
  if (val.length) insights.push('从第 ' + number(val[0].step) + ' 步到第 ' + number(val.at(-1).step) + ' 步，P4 验证交叉熵由 ' + decimal(val[0].validation_cross_entropy) + ' 变为 ' + decimal(val.at(-1).validation_cross_entropy) + '。固定训练批联合损失由 ' + decimal(r.fixed_train_batch_loss_before) + ' 变为 ' + decimal(r.fixed_train_batch_loss_after) + '。');
  for (const g of report.groups) {
    const n = g.samples.length, m = g.metrics;
    insights.push(splitName(m.split) + '（生成上限 ' + number(m.max_new_tokens) + '）共 ' + n + ' 条。模型 ROUGE-L 为 ' + decimal(m.whitespace_rouge_l_f1) + '，复制原句为 ' + decimal(m.copy_baseline?.whitespace_rouge_l_f1) + '。逐条比较时有 ' + g.below + ' 条低于基线，' + g.zeros + ' 条模型得分为零。');
  }
  if (!insights.length) insights.push('当前目录没有验证记录或逐条评估。');
  byId('findings').replaceChildren(...insights.map(s => el('p',s)));
  for (const g of report.groups) {
    if (!g.below) continue;
    const worst = g.samples.reduce((a,b) => a.baseline - a.rouge >= b.baseline - b.rouge ? a : b);
    const example = el('button',splitName(g.metrics.split) + ' · 查看与基线差距最大的样本 #' + padIndex(worst.index));
    example.type = 'button';
    example.addEventListener('click',() => {
      selectGroup(g); location.hash = '#compare'; switchView(); selected = worst; showSample(true);
    });
    byId('findings').append(example);
  }
  byId('validation-table').replaceChildren(...val.map(p => tableRow([number(p.step),decimal(p.fixed_train_batch_loss),decimal(p.validation_cross_entropy_by_depth?.medium),decimal(p.validation_cross_entropy)])));
}

function switchView() {
  const training = location.hash === '#training';
  byId('compare').hidden = training; byId('training').hidden = !training;
  for (const view of ['compare','training']) {
    const link = byId('nav-' + view);
    if ((view === 'training') === training) link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  }
}

function display(raw) {
  const warnings = [...(raw.warnings || [])], groups = [];
  for (const input of raw.evaluations) {
    try { groups.push(analyzeGroup(input)); }
    catch (error) { warnings.push(input.name + ' · ' + error.message); }
  }
  groups.sort((a,b) => Number(b.metrics.split === 'test') - Number(a.metrics.split === 'test') || a.name.localeCompare(b.name, 'en', {numeric:true}));
  report = {name:raw.name, result:raw.result, training:raw.training || [], validation:raw.validation || [], groups};
  for (const elem of document.querySelectorAll('.runtag')) {
    elem.textContent = report.name + '  /  检查点 ' + number(report.result.selected_step) + ' 步';
  }
  byId('empty').hidden = true; byId('report').hidden = false;
  const buttons = groups.map(g => {
    const button = el('button',splitName(g.metrics.split) + ' ' + g.samples.length);
    if (groups.some(k => k !== g && k.metrics.split === g.metrics.split)) button.append(' · ' + g.name);
    button.type = 'button'; button.dataset.name = g.name;
    button.addEventListener('click', () => selectGroup(g));
    return button;
  });
  byId('split-switch').replaceChildren(...buttons);
  group = undefined; selected = undefined; visible = [];
  byId('no-eval').hidden = Boolean(groups.length);
  byId('eval-content').hidden = !groups.length;
  if (groups.length) selectGroup(groups[0]);
  drawTraining(); showNotice(warnings.join('\n')); switchView();
}

async function readDirectory(fileList) {
  const files = new Map(), warnings = [];
  let name = '';
  for (const file of fileList) {
    const segments = file.webkitRelativePath.split('/');
    name = segments.shift();
    files.set(segments.join('/'), file);
  }
  const done = files.get('result.json');
  if (!done) throw new Error('未找到训练完成记录 result.json，请选择本次训练目录。');
  const result = JSON.parse(await done.text());
  if (!result.fuse_stage_outputs || !Number.isFinite(result.steps)) throw new Error('请选择包含 P4 融合输出的已完成 LexiFAR 训练。');
  async function readLog(path) {
    if (!files.has(path)) { warnings.push('未找到 ' + path + '。'); return []; }
    const log = JSON.parse(await files.get(path).text());
    if (!Array.isArray(log)) throw new Error(path + ' 的格式有误。');
    return log;
  }
  const [training,validation] = await Promise.all([readLog('training_log.json'),readLog('validation_log.json')]);
  const trainedAt = Math.max(done.lastModified, files.get('weights.pt')?.lastModified || 0);
  const folders = new Set();
  for (const path of files.keys()) {
    const match = /^evaluation\/([^/]+)\/(?:metrics\.json|predictions\.jsonl)$/u.exec(path);
    if (match) folders.add(match[1]);
  }
  const evaluations = [];
  for (const folder of folders) {
    const prefix = 'evaluation/' + folder + '/', metrics = files.get(prefix + 'metrics.json'), preds = files.get(prefix + 'predictions.jsonl');
    if (!metrics || !preds) { warnings.push(folder + ' 缺少成对的指标或预测文件。'); continue; }
    if (Math.min(metrics.lastModified,preds.lastModified) < trainedAt) { warnings.push(folder + ' 的评估文件早于当前训练完成记录或权重，已跳过。'); continue; }
    try {
      const m = JSON.parse(await metrics.text());
      const rows = (await preds.text()).split(/\r?\n/u).filter(s => s.trim()).map(s => JSON.parse(s));
      evaluations.push({name:folder, metrics:m, rows});
    } catch (error) { warnings.push(folder + ' 无法读取 · ' + error.message); }
  }
  return {name,result,training,validation,evaluations,warnings};
}

byId('load').addEventListener('click', () => byId('folder').click());
byId('start').addEventListener('click', () => byId('folder').click());
byId('folder').addEventListener('change', async event => {
  if (!event.target.files.length) return;
  const version = ++importVersion;
  showNotice('正在读取本地评估文件…');
  try { const data = await readDirectory(event.target.files); if (version === importVersion) display(data); }
  catch (error) { if (version === importVersion) showNotice(error.message); }
  finally { if (version === importVersion) event.target.value = ''; }
});
byId('filter').addEventListener('change',filterSamples);
byId('search').addEventListener('input',filterSamples);
byId('diff').addEventListener('change',() => { if (selected) showTexts(); });
byId('clear').addEventListener('click',() => { byId('filter').value='all'; byId('search').value=''; filterSamples(); });
for (const [id,delta] of [['prev',-1],['next',1]]) {
  byId(id).addEventListener('click',() => {
    const next = visible[visible.indexOf(selected) + delta];
    if (next) { selected = next; showSample(true); }
  });
}
const narrowScreen = window.matchMedia('(max-width: 650px)');
function fitIndex() { byId('index-toggle').open = !narrowScreen.matches; }
narrowScreen.addEventListener('change',fitIndex);
fitIndex();
window.addEventListener('hashchange',() => { switchView(); window.scrollTo(0,0); });
try {
  const saved = JSON.parse(byId('report-data').textContent);
  if (saved) display(saved); else switchView();
} catch (error) { showNotice('无法载入报告数据 · ' + error.message); }
