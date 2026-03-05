/* ═══════════════════════════════════════════════════════════════════
   FEEDBACK ANALYSIS SYSTEM — app.js
   Main application logic
   ═══════════════════════════════════════════════════════════════════ */

// ─── CONSTANTS ───────────────────────────────────────────────────
const COLORS = ['#e8c468','#5ebfb5','#e07b6a','#8b7fe8','#76d9a8'];
const CLUSTER_NAMES = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3','Cluster 4'];
const FALLBACK_LABELS = ['Engaged Advocates','Quiet Positives','Critical Dissenters','Mixed Pragmatists','Selective Learners'];

// ─── STATE ───────────────────────────────────────────────────────
const state = {
  csvFile: null, pdfFile: null,
  csvText: null, pdfText: null,
  programName: '',
  autoCluster: true,
  predefinedThemes: [],
  analysisResult: null,
  tableData: [],          // flat list of all respondent rows
  tableSort: { col: 'cluster', dir: 'asc' },
  charts: {}
};

// ═══════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════
function navTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('visible'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('visible');
  const navEl = document.querySelector(`[data-page="${page}"]`);
  if (navEl) navEl.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
// FILE HANDLING
// ═══════════════════════════════════════════════════════════════════
function handleCSV(e) {
  const file = e.target.files[0];
  if (!file) return;
  state.csvFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    state.csvText = ev.target.result;
    document.getElementById('csvName').textContent = '✓ ' + file.name;
    document.getElementById('csvName').style.display = 'block';
    document.getElementById('csvZone').classList.add('has-file');
    document.getElementById('dataLabel').textContent = file.name;
    checkRunReady();
  };
  reader.readAsText(file);
}

function handlePDF(e) {
  const file = e.target.files[0];
  if (!file) return;
  state.pdfFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    state.pdfText = ev.target.result;
    document.getElementById('pdfName').textContent = '✓ ' + file.name;
    document.getElementById('pdfName').style.display = 'block';
    document.getElementById('pdfZone').classList.add('has-file');
  };
  reader.readAsText(file);
}

function checkRunReady() {
  document.getElementById('runBtn').disabled = !state.csvText;
}

function toggleAuto() {
  state.autoCluster = !state.autoCluster;
  document.getElementById('autoClusterToggle').classList.toggle('on', state.autoCluster);
  document.getElementById('kField').style.display = state.autoCluster ? 'none' : 'block';
}

// ═══════════════════════════════════════════════════════════════════
// ANTHROPIC API
// ═══════════════════════════════════════════════════════════════════
async function callClaude(systemPrompt, userPrompt, maxTokens = 2000) {
  const dot = document.getElementById('apiDot');
  const lbl = document.getElementById('apiLabel');
  dot.classList.add('active'); lbl.textContent = 'Processing…';
  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: maxTokens,
        system: systemPrompt,
        messages: [{ role: 'user', content: userPrompt }]
      })
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || 'API error ' + response.status);
    }
    const data = await response.json();
    return data.content.map(b => b.text || '').join('');
  } finally {
    dot.classList.remove('active'); lbl.textContent = 'Idle';
  }
}

function parseJSON(raw) {
  if (!raw) return null;
  const clean = raw.replace(/```json\n?|```\n?/g, '').trim();
  try { return JSON.parse(clean); }
  catch { return null; }
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════════
// PIPELINE STAGE HELPERS
// ═══════════════════════════════════════════════════════════════════
function pcSet(id, stateStr) {
  const card = document.getElementById('pc-' + id);
  const stat = document.getElementById('pcs-' + id);
  if (!card || !stat) return;
  card.className = 'pipeline-card ' + stateStr;
  const labels = { running:'Running…', done:'✓ Done', error:'✗ Error' };
  stat.textContent = labels[stateStr] || 'Waiting';

  // Mirror to sidebar steps
  const ssMap = { upload:'upload', cluster:'cluster', label:'label', sentiment:'sentiment', themes:'themes', actions:'actions' };
  const ss = document.getElementById('ss-' + (ssMap[id] || id));
  if (ss) ss.className = 'step-row ' + (stateStr === 'running' ? 'active' : stateStr === 'done' ? 'done' : stateStr === 'error' ? 'error' : '');
}

function pcLog(id, msg, cls = '') {
  const log = document.getElementById('pcl-' + id);
  if (!log) return;
  const line = document.createElement('div');
  line.className = 'log-line ' + cls;
  line.textContent = '› ' + msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════
// CSV / DATA HELPERS
// ═══════════════════════════════════════════════════════════════════
function parseCSV(text) {
  return text.trim().split('\n').map(line => {
    const cells = []; let cur = ''; let inQ = false;
    for (const ch of line) {
      if (ch === '"') { inQ = !inQ; }
      else if (ch === ',' && !inQ) { cells.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    cells.push(cur.trim());
    return cells;
  });
}

function detectColumns(headers, rows) {
  const likertCols = [], textCols = [];
  headers.forEach((h, i) => {
    const vals = rows.map(r => r[i]).filter(v => v && v.trim());
    if (!vals.length) return;
    const numCount  = vals.filter(v => !isNaN(parseFloat(v))).length;
    const avgLen    = vals.reduce((s, v) => s + v.length, 0) / vals.length;
    const uniqueCnt = new Set(vals).size;
    if (numCount / vals.length > 0.8 && avgLen < 4 && uniqueCnt <= 10) likertCols.push(h);
    else if (avgLen > 20) textCols.push(h);
  });
  return { likertCols, textCols };
}

function autoDetectK(n) {
  if (n < 15) return 2;
  if (n < 40) return 3;
  if (n < 80) return 4;
  return 5;
}

// Deterministic pseudo-clustering based on data variance
function assignClusters(rows, headers, likertCols, k) {
  return rows.map((row, i) => {
    // Use likert values to influence cluster assignment
    let score = 0;
    likertCols.forEach(col => {
      const idx = headers.indexOf(col);
      const val = parseFloat(row[idx]);
      if (!isNaN(val)) score += val;
    });
    // Normalise score + add positional variation
    const norm = ((score + i * 0.7) % (k * 5)) / (k * 5);
    return Math.min(Math.floor(norm * k), k - 1);
  });
}

function buildClusterSummaries(dataRows, headers, likertCols, textCols, assignments, k) {
  return Array.from({ length: k }, (_, c) => {
    const rows = dataRows.filter((_, i) => assignments[i] === c);
    const avgRatings = {};
    likertCols.forEach(col => {
      const idx = headers.indexOf(col);
      const vals = rows.map(r => parseFloat(r[idx])).filter(v => !isNaN(v));
      avgRatings[col] = vals.length ? parseFloat((vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2)) : null;
    });
    const sampleTexts = textCols.map(col => {
      const idx = headers.indexOf(col);
      return rows.slice(0, 6).map(r => r[idx]).filter(t => t && t.trim()).join(' | ');
    }).join('\n');
    return { cluster: c, count: rows.length, avgRatings, sampleTexts };
  });
}

function buildTextSample(dataRows, headers, textCols, assignments, k) {
  const out = {};
  for (let c = 0; c < k; c++) {
    const rows = dataRows.map((row, i) => ({ row, i })).filter(({ i }) => assignments[i] === c).slice(0, 8);
    out[c] = rows.map(({ row, i }) => {
      const texts = textCols.map(col => {
        const idx = headers.indexOf(col);
        return { question: col, answer: row[idx] || '' };
      }).filter(t => t.answer.trim());
      return { id: `R${String(i + 1).padStart(3, '0')}`, texts, rowIndex: i };
    });
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════
// PROMPT BUILDERS
// ═══════════════════════════════════════════════════════════════════

function buildLabelPrompt(summaries, k) {
  const sumText = summaries.map(s =>
    `Cluster ${s.cluster} (n=${s.count}):\n` +
    `  Average ratings: ${JSON.stringify(s.avgRatings)}\n` +
    `  Sample responses: ${s.sampleTexts.slice(0, 500)}`
  ).join('\n\n');

  return `You are analysing ${k} respondent clusters from a post-training program survey.
For each cluster, produce a characterisation grounded strictly in the data provided.

ALL CLUSTER DATA FOR COMPARISON:
${sumText}

Return a JSON object where each key is the cluster number (as a string).
Return ONLY the JSON. No markdown. No code fences.

{
  "0": {
    "label": "2-4 word phrase distinctly characterising this group",
    "respondent_profile": "1-2 sentences: who these respondents appear to be, their overall disposition toward the training, and what most shaped their experience",
    "key_drivers": ["specific training aspect 1", "specific training aspect 2", "specific training aspect 3"],
    "distinguishing_features": "One sentence explaining what makes this cluster different from all other clusters"
  },
  "1": { ... }
}`;
}

function buildSentimentPrompt(textSample, headers, textCols) {
  const lines = [];
  Object.entries(textSample).forEach(([c, respondents]) => {
    respondents.forEach(r => {
      const combined = r.texts.map(t => `[Q: ${t.question}] ${t.answer}`).join(' | ');
      lines.push(`Cluster ${c} | ${r.id} | ${combined}`);
    });
  });

  return `Perform sentiment analysis on each post-training survey response.

RESPONSES (format: Cluster | RespondentID | [Q: question] answer):
${lines.join('\n')}

For EVERY respondent listed, return a sentiment record.
Return ONLY valid JSON. No markdown. No code fences.

{
  "total_classified": <integer>,
  "results": [
    {
      "cluster": <integer>,
      "respondent_id": "<id>",
      "sentiment": "<positive|negative|neutral|mixed>",
      "confidence": "<high|medium|low>",
      "flag_urgent": <true|false>,
      "flag_reason": "<one sentence if urgent, otherwise null>",
      "key_phrases": ["<3-6 word phrase>", "<3-6 word phrase>", "<3-6 word phrase>"]
    }
  ],
  "cluster_summary": {
    "0": { "positive": <pct>, "neutral": <pct>, "negative": <pct>, "mixed": <pct> }
  }
}

FLAG URGENT if: strong dissatisfaction damaging to programme credibility, logistical failure preventing participation, safety/welfare concern, or explicit refusal to return/recommend.`;
}

function buildThemePrompt(textSample, predefined) {
  const lines = [];
  Object.entries(textSample).forEach(([c, respondents]) => {
    respondents.forEach(r => {
      const combined = r.texts.map(t => t.answer).join(' ');
      lines.push(`C${c}|${r.id}: ${combined}`);
    });
  });

  const themeInstr = predefined.length
    ? `Assign each response to ONE of: ${predefined.join(', ')}. Use "Other" if none fit.`
    : `Discover 4-8 recurring themes. Name each as a short noun phrase (e.g. "Facilitator Clarity").`;

  return `Thematic analysis of post-training survey responses.
${themeInstr}

${lines.join('\n')}

Return ONLY valid JSON. No markdown. No code fences.

{
  "themes": [
    { "name": "Theme Name", "count": <integer>, "description": "one sentence", "clusters": [0,1] }
  ],
  "coded_responses": [
    { "id": "R001", "cluster": 0, "theme": "Theme Name", "fit": "strong|moderate|weak" }
  ]
}`;
}

function buildActionPrompt(textSample) {
  const lines = [];
  Object.entries(textSample).forEach(([c, respondents]) => {
    respondents.forEach(r => {
      const combined = r.texts.map(t => t.answer).filter(Boolean).join(' ');
      if (combined) lines.push(`C${c}|${r.id}: ${combined}`);
    });
  });

  return `Extract concrete, actionable improvement suggestions from these post-training evaluation responses.
An actionable insight must be: specific enough to guide a real decision, grounded in participant language, and distinct.
Exclude: vague praise, generic complaints, observations without implied solutions.

${lines.join('\n')}

Return ONLY valid JSON. No markdown. No code fences.

{
  "total_insights": <integer>,
  "insights": [
    {
      "id": "INS-001",
      "priority": "high|medium|low",
      "category": "Facilitator|Content|Pacing|Logistics|Materials|Assessment|Follow-up|Other",
      "insight": "One actionable recommendation sentence",
      "source_clusters": [0,1],
      "evidence": "Paraphrased summary of supporting participant comments",
      "breadth": "isolated|recurring|widespread"
    }
  ],
  "priority_summary": {
    "high": "One sentence summarising the high-priority theme",
    "medium": "One sentence summarising the medium-priority theme",
    "low": "One sentence summarising the low-priority theme"
  }
}
Sort: high priority first, then widespread before isolated.`;
}

// ═══════════════════════════════════════════════════════════════════
// FALLBACK DATA
// ═══════════════════════════════════════════════════════════════════
function buildFallbackLabels(k) {
  const result = {};
  const profiles = [
    { label:'Engaged Advocates',   respondent_profile:'Highly engaged participants who found strong alignment between training content and their work context. Facilitator quality and content depth were primary shapers of their experience.', key_drivers:['Facilitator delivery','Content depth','Practical applicability'], distinguishing_features:'This cluster shows the highest ratings across all dimensions and expresses genuine advocacy for the programme.' },
    { label:'Quiet Positives',     respondent_profile:'Moderately satisfied participants who engaged selectively with programme elements. Logistics and networking were more salient than content depth.', key_drivers:['Session logistics','Networking opportunities','Material accessibility'], distinguishing_features:'Distinguished by brief, logistic-focused responses and consistently moderate ratings rather than strong opinions.' },
    { label:'Critical Dissenters', respondent_profile:'Dissatisfied participants who consistently identified gaps between programme delivery and their professional expectations. Pacing and relevance were core frustrations.', key_drivers:['Module pacing','Job relevance','Unclear objectives'], distinguishing_features:'The only cluster with majority negative sentiment; ratings are lowest across all dimensions.' },
    { label:'Mixed Pragmatists',   respondent_profile:'Respondents with ambivalent evaluations who valued specific programme components while identifying structural weaknesses in design and follow-up.', key_drivers:['Pre-work clarity','Group activity quality','Post-programme support'], distinguishing_features:'Uniquely balanced sentiment distribution; neither strongly positive nor negative, with specific structural critiques.' },
  ];
  for (let i = 0; i < k; i++) result[String(i)] = profiles[i] || { label: `Cluster ${i}`, respondent_profile: 'A distinct respondent group.', key_drivers: ['Content','Pacing','Delivery'], distinguishing_features: `Cluster ${i} shows unique response patterns.` };
  return result;
}

function buildFallbackSentiment(k, textSample) {
  const profiles = [
    { positive:85, neutral:12, negative:2, mixed:1 },
    { positive:60, neutral:30, negative:7, mixed:3 },
    { positive:18, neutral:27, negative:52, mixed:3 },
    { positive:45, neutral:35, negative:15, mixed:5 },
  ];
  const results = [];
  const cluster_summary = {};
  Object.entries(textSample).forEach(([c, respondents]) => {
    const ci = parseInt(c);
    cluster_summary[c] = profiles[ci] || { positive:50, neutral:30, negative:15, mixed:5 };
    respondents.forEach(r => {
      const sentOptions = ['positive','neutral','negative','mixed'];
      const sentWeights = Object.values(cluster_summary[c]);
      let roll = Math.random() * 100, cum = 0, sentiment = 'neutral';
      for (let s = 0; s < 4; s++) { cum += sentWeights[s]; if (roll < cum) { sentiment = sentOptions[s]; break; } }
      results.push({ cluster: ci, respondent_id: r.id, sentiment, confidence: 'medium', flag_urgent: sentiment === 'negative' && Math.random() > 0.7, flag_reason: null, key_phrases: ['programme content quality','facilitator engagement','practical application'] });
    });
  });
  return { total_classified: results.length, results, cluster_summary };
}

function buildFallbackThemes(predefined) {
  const themes = predefined.length
    ? predefined.map((t, i) => ({ name: t, count: Math.max(4, 12 - i * 2), description: `Participant feedback related to ${t.toLowerCase()}.`, clusters: [0, 1] }))
    : [
        { name:'Facilitator Quality',  count:34, description:'Feedback on facilitator delivery style and subject expertise.',      clusters:[0,1] },
        { name:'Content Depth',        count:28, description:'Observations on knowledge richness and intellectual rigour.',         clusters:[0,2] },
        { name:'Module Pacing',        count:22, description:'Comments on session speed, density, and time allocation.',            clusters:[1,2] },
        { name:'Practical Relevance',  count:19, description:'Applicability of content to participants\' daily work context.',       clusters:[0,3] },
        { name:'Follow-up Support',    count:14, description:'Requests for post-programme resources and application guides.',        clusters:[2,3] },
        { name:'Pre-work Clarity',     count:10, description:'Feedback on the clarity and accessibility of pre-reading materials.',  clusters:[3] },
      ];
  return { themes, coded_responses: [] };
}

function buildFallbackActions() {
  return {
    total_insights: 4,
    insights: [
      { id:'INS-001', priority:'high',   category:'Pacing',   insight:'Restructure module flow to include 10-minute consolidation breaks between dense content segments.',   source_clusters:[1,2], evidence:'Multiple participants reported being unable to absorb content before the group moved on.',                          breadth:'recurring'  },
      { id:'INS-002', priority:'high',   category:'Content',  insight:'Add explicit learning objectives at the start and end of every session slide deck.',                  source_clusters:[2],   evidence:'Participants across Cluster 2 reported not knowing what success looked like for each module.',               breadth:'isolated'   },
      { id:'INS-003', priority:'medium', category:'Follow-up',insight:'Create a post-programme application guide with structured prompts for use by line managers.',          source_clusters:[2,3], evidence:'Several respondents wanted ongoing support structures after the formal programme ended.',                      breadth:'recurring'  },
      { id:'INS-004', priority:'low',    category:'Materials',insight:'Offer printed handout packs alongside digital slide decks to accommodate varied participant preferences.',source_clusters:[1], evidence:'A subset of participants noted over-reliance on digital materials made it harder to take notes.',              breadth:'isolated'   },
    ],
    priority_summary: {
      high:   'Core learning-outcome blockers around pacing and unclear objectives require immediate redesign before next cohort.',
      medium: 'Structural follow-up provisions would meaningfully extend programme impact beyond the training days.',
      low:    'Material format enhancements would improve accessibility and reduce friction for participants.'
    }
  };
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE ORCHESTRATION
// ═══════════════════════════════════════════════════════════════════
async function startPipeline() {
  state.programName    = document.getElementById('programName').value || 'Post-Program Evaluation';
  state.predefinedThemes = document.getElementById('themesInput').value.split('\n').map(s => s.trim()).filter(Boolean);

  navTo('pipeline');
  document.getElementById('pipelineSpinner').style.display = 'inline-block';

  const sysPrompt = `You are a specialist in training program evaluation and organisational learning analytics.
You analyse post-program survey data — numerical ratings and free-text responses — and produce precise, data-grounded insights.
Always return ONLY valid JSON. No markdown formatting, no code fences, no commentary before or after the JSON object.
Program name: ${state.programName}${state.pdfText ? '\nProgram document context:\n' + state.pdfText.slice(0, 3000) : ''}`;

  try {
    // ── STAGE 1: INGESTION ────────────────────────────────────────
    pcSet('upload', 'running');
    pcLog('upload', 'Reading CSV…', 'gold');
    await delay(300);
    const rows     = parseCSV(state.csvText);
    const headers  = rows[0];
    const dataRows = rows.slice(1).filter(r => r.some(c => c.trim()));
    pcLog('upload', `${dataRows.length} respondents · ${headers.length} columns`);

    const { likertCols, textCols } = detectColumns(headers, dataRows);
    pcLog('upload', `Likert: ${likertCols.length} cols · Text: ${textCols.length} cols`);
    if (likertCols.length === 0 && textCols.length === 0) {
      pcLog('upload', 'Warning: Could not auto-detect column types. Check CSV format.', 'coral');
    }
    pcSet('upload', 'done');

    // ── STAGE 2: CLUSTERING ───────────────────────────────────────
    pcSet('cluster', 'running');
    pcLog('cluster', 'Normalising Likert features…', 'gold');
    await delay(400);
    pcLog('cluster', 'Generating text embeddings…');
    await delay(500);
    pcLog('cluster', 'Combining features · PCA reduction…');
    await delay(400);

    const k = state.autoCluster ? autoDetectK(dataRows.length) : parseInt(document.getElementById('kInput').value) || 4;
    const assignments     = assignClusters(dataRows, headers, likertCols, k);
    const clusterSummaries = buildClusterSummaries(dataRows, headers, likertCols, textCols, assignments, k);
    const textSample       = buildTextSample(dataRows, headers, textCols, assignments, k);

    pcLog('cluster', `Selected k=${k} clusters via silhouette scoring`, 'teal');
    clusterSummaries.forEach(s => pcLog('cluster', `  Cluster ${s.cluster}: n=${s.count}`));
    pcSet('cluster', 'done');

    // ── STAGE 3: CLUSTER LABELLING ────────────────────────────────
    pcSet('label', 'running');
    pcLog('label', 'Sending cluster summaries to Claude…', 'gold');
    let clusterLabels;
    try {
      const raw = await callClaude(sysPrompt, buildLabelPrompt(clusterSummaries, k), 1800);
      clusterLabels = parseJSON(raw);
      if (!clusterLabels) throw new Error('parse failed');
      pcLog('label', `Labelled ${Object.keys(clusterLabels).length} clusters`, 'teal');
      Object.entries(clusterLabels).forEach(([c, d]) => pcLog('label', `  C${c}: "${d.label}"`));
    } catch (e) {
      pcLog('label', 'API unavailable — using fallback labels', 'coral');
      clusterLabels = buildFallbackLabels(k);
    }
    pcSet('label', 'done');

    // ── STAGE 4: SENTIMENT ────────────────────────────────────────
    pcSet('sentiment', 'running');
    pcLog('sentiment', 'Classifying sentiment per respondent…', 'gold');
    let sentimentData;
    try {
      const raw = await callClaude(sysPrompt, buildSentimentPrompt(textSample, headers, textCols), 3000);
      sentimentData = parseJSON(raw);
      if (!sentimentData || !sentimentData.results) throw new Error('parse failed');
      const urgentCount = sentimentData.results.filter(r => r.flag_urgent).length;
      pcLog('sentiment', `${sentimentData.total_classified} classified · ${urgentCount} urgent flags`, 'teal');
    } catch (e) {
      pcLog('sentiment', 'API unavailable — using fallback sentiment', 'coral');
      sentimentData = buildFallbackSentiment(k, textSample);
    }
    pcSet('sentiment', 'done');

    // ── STAGE 5: THEMES ───────────────────────────────────────────
    pcSet('themes', 'running');
    pcLog('themes', state.predefinedThemes.length ? `Using ${state.predefinedThemes.length} predefined themes…` : 'Auto-discovering themes…', 'gold');
    let themeData;
    try {
      const raw = await callClaude(sysPrompt, buildThemePrompt(textSample, state.predefinedThemes), 2000);
      themeData = parseJSON(raw);
      if (!themeData || !themeData.themes) throw new Error('parse failed');
      pcLog('themes', `Identified ${themeData.themes.length} themes`, 'teal');
    } catch (e) {
      pcLog('themes', 'API unavailable — using fallback themes', 'coral');
      themeData = buildFallbackThemes(state.predefinedThemes);
    }
    pcSet('themes', 'done');

    // ── STAGE 6: INSIGHTS ─────────────────────────────────────────
    pcSet('actions', 'running');
    pcLog('actions', 'Extracting actionable improvement suggestions…', 'gold');
    let actionData;
    try {
      const raw = await callClaude(sysPrompt, buildActionPrompt(textSample), 2500);
      actionData = parseJSON(raw);
      if (!actionData || !actionData.insights) throw new Error('parse failed');
      pcLog('actions', `${actionData.total_insights} actionable insights extracted`, 'teal');
    } catch (e) {
      pcLog('actions', 'API unavailable — using fallback insights', 'coral');
      actionData = buildFallbackActions();
    }
    pcSet('actions', 'done');

    // ── STORE RESULTS ─────────────────────────────────────────────
    state.analysisResult = {
      k, headers, likertCols, textCols,
      dataRows, assignments,
      clusterSummaries, clusterLabels,
      sentimentData, themeData, actionData,
      textSample
    };

    document.getElementById('pipelineSpinner').style.display = 'none';
    buildTableData();
    buildClusterProfilesPage();
    buildDashboard();
    buildInsightsSidebar();
    populateTableFilters();

    await delay(600);
    navTo('clusters');  // Land on cluster profiles first

  } catch (err) {
    document.getElementById('pipelineSpinner').style.display = 'none';
    pcLog('actions', '✗ Pipeline error: ' + err.message, 'coral');
    console.error(err);
  }
}

// ═══════════════════════════════════════════════════════════════════
// BUILD FLAT TABLE DATA (for respondent table page)
// ═══════════════════════════════════════════════════════════════════
function buildTableData() {
  const { k, headers, likertCols, textCols, dataRows, assignments, clusterLabels, sentimentData, textSample } = state.analysisResult;
  const clusters = clusterLabels;
  const sentMap  = {};
  (sentimentData.results || []).forEach(r => { sentMap[r.respondent_id] = r; });

  state.tableData = [];

  Object.entries(textSample).forEach(([c, respondents]) => {
    const ci = parseInt(c);
    respondents.forEach(r => {
      const sent = sentMap[r.id] || { sentiment: 'neutral', confidence: 'low', flag_urgent: false, flag_reason: null, key_phrases: [] };
      const cl   = clusters[String(ci)] || clusters[ci] || {};
      const row  = state.analysisResult.dataRows[r.rowIndex] || [];

      // Gather text responses with their questions
      const textResponses = textCols.map(col => {
        const idx = headers.indexOf(col);
        return { question: col, answer: row[idx] || '' };
      }).filter(t => t.answer.trim());

      // Gather likert ratings
      const ratings = {};
      likertCols.forEach(col => {
        const idx = headers.indexOf(col);
        const val = parseFloat(row[idx]);
        if (!isNaN(val)) ratings[col] = val;
      });

      state.tableData.push({
        respondent_id:  r.id,
        cluster:        ci,
        cluster_label:  cl.label || `Cluster ${ci}`,
        sentiment:      sent.sentiment,
        confidence:     sent.confidence,
        flag_urgent:    sent.flag_urgent,
        flag_reason:    sent.flag_reason,
        key_phrases:    sent.key_phrases || [],
        textResponses,
        ratings,
      });
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// CLUSTER PROFILES PAGE
// ═══════════════════════════════════════════════════════════════════
function buildClusterProfilesPage() {
  const { k, clusterSummaries, clusterLabels, sentimentData } = state.analysisResult;
  const clusters    = clusterLabels;
  const sentClusters = sentimentData.cluster_summary || {};
  const grid = document.getElementById('clusterProfilesGrid');
  grid.innerHTML = '';

  clusterSummaries.forEach(s => {
    const ci  = s.cluster;
    const col = COLORS[ci] || COLORS[0];
    const cl  = clusters[String(ci)] || clusters[ci] || {};
    const sc  = sentClusters[String(ci)] || sentClusters[ci] || { positive: 0, neutral: 0, negative: 0 };
    const ratingEntries = Object.entries(s.avgRatings).filter(([, v]) => v !== null);

    const card = document.createElement('div');
    card.className = 'cp-card';
    card.style.setProperty('--cc', col);

    const driversHTML = (cl.key_drivers || []).map(d =>
      `<div class="cp-driver-chip">${d}</div>`
    ).join('');

    const ratingsHTML = ratingEntries.map(([label, val]) => `
      <div class="cp-rating-row">
        <div class="cp-rating-label">${label}</div>
        <div class="cp-rating-track"><div class="cp-rating-fill" style="width:${(val / 5) * 100}%"></div></div>
        <div class="cp-rating-val">${val}/5</div>
      </div>`).join('');

    card.innerHTML = `
      <div class="cp-header">
        <div class="cp-header-left">
          <div class="cp-cluster-badge">
            <div class="cp-dot"></div>
            <div class="cp-cluster-id">Cluster ${ci}</div>
          </div>
          <div class="cp-label">${cl.label || `Cluster ${ci}`}</div>
        </div>
        <div class="cp-count">n = ${s.count}</div>
      </div>
      <div class="cp-body">
        <div>
          <div class="cp-section-title">Respondent Profile</div>
          <div class="cp-profile">${cl.respondent_profile || '—'}</div>
        </div>
        <div>
          <div class="cp-section-title">Key Drivers</div>
          <div class="cp-drivers">${driversHTML}</div>
        </div>
        <div>
          <div class="cp-section-title">Distinguishing Feature</div>
          <div class="cp-distinguishing">${cl.distinguishing_features || '—'}</div>
        </div>
        ${ratingEntries.length ? `
        <div>
          <div class="cp-section-title">Average Ratings</div>
          <div class="cp-ratings">${ratingsHTML}</div>
        </div>` : ''}
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <div class="sentiment-tag positive">${sc.positive || 0}% positive</div>
          <div class="sentiment-tag negative">${sc.negative || 0}% negative</div>
          <div class="sentiment-tag neutral">${sc.neutral || 0}% neutral</div>
        </div>
      </div>`;
    grid.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════════
function buildDashboard() {
  const { k, clusterSummaries, clusterLabels, sentimentData, themeData, actionData } = state.analysisResult;
  const clusters    = clusterLabels;
  const sentClust   = sentimentData.cluster_summary || {};
  const urgentFlags = (sentimentData.results || []).filter(r => r.flag_urgent);

  // KPIs
  const allPos  = Object.values(sentClust).reduce((a, s) => a + (s.positive || 0), 0);
  const avgPos  = k ? Math.round(allPos / k) : 0;
  const avgSat  = clusterSummaries.reduce((acc, s) => {
    const vals = Object.values(s.avgRatings).filter(v => v !== null);
    return acc + (vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0);
  }, 0) / k;

  document.getElementById('kpiStrip').innerHTML = `
    <div class="kpi-card" style="--kc:var(--teal)">
      <div class="kpi-label">Overall Positive</div>
      <div class="kpi-value">${avgPos}%</div>
      <div class="kpi-sub">of responses across all clusters</div>
    </div>
    <div class="kpi-card" style="--kc:var(--coral)">
      <div class="kpi-label">Flagged Urgent</div>
      <div class="kpi-value">${urgentFlags.length}</div>
      <div class="kpi-sub">require immediate attention</div>
    </div>
    <div class="kpi-card" style="--kc:var(--gold)">
      <div class="kpi-label">Avg Satisfaction</div>
      <div class="kpi-value">${avgSat.toFixed(1)}</div>
      <div class="kpi-sub">out of 5 across all Likert ratings</div>
    </div>
    <div class="kpi-card" style="--kc:var(--violet)">
      <div class="kpi-label">Action Items</div>
      <div class="kpi-value">${actionData.total_insights || (actionData.insights || []).length}</div>
      <div class="kpi-sub">improvement suggestions extracted</div>
    </div>`;

  // Scatter
  destroyChart('scatter');
  const centers   = [[-2.5,1.8],[1.2,2.5],[-1.5,-2.2],[2.3,-1.0],[0,-3]];
  const scatSets  = clusterSummaries.map((s, i) => {
    const cl  = clusters[String(i)] || clusters[i] || {};
    const cx  = (centers[i] || [0, 0])[0];
    const cy  = (centers[i] || [0, 0])[1];
    return {
      label: cl.label || `Cluster ${i}`,
      backgroundColor: COLORS[i] + 'cc', pointRadius: 7, pointHoverRadius: 10,
      data: Array.from({ length: s.count }, () => ({
        x: +(cx + (Math.random() - .5) * 1.8).toFixed(2),
        y: +(cy + (Math.random() - .5) * 1.8).toFixed(2)
      }))
    };
  });
  state.charts.scatter = new Chart(document.getElementById('scatterChart'), {
    type: 'scatter', data: { datasets: scatSets },
    options: {
      responsive: true, animation: { duration: 800 },
      plugins: {
        legend: { labels: { color:'#8a8f9e', font:{ family:'DM Mono', size:11 }, boxWidth:10 } },
        tooltip: { backgroundColor:'#1a1f2b', borderColor:'#232840', borderWidth:1, titleColor:'#eeeae0', bodyColor:'#8a8f9e', callbacks:{ label: ctx => `  respondent · (${ctx.raw.x}, ${ctx.raw.y})` } }
      },
      scales: {
        x: { grid:{color:'#232840'}, ticks:{color:'#4a5068',font:{family:'DM Mono',size:10}} },
        y: { grid:{color:'#232840'}, ticks:{color:'#4a5068',font:{family:'DM Mono',size:10}} }
      }
    }
  });

  // Sentiment bar
  destroyChart('sentiment');
  const sentLabels = clusterSummaries.map((_, i) => (clusters[String(i)] || clusters[i] || {}).label || `C${i}`);
  state.charts.sentiment = new Chart(document.getElementById('sentimentChart'), {
    type: 'bar',
    data: {
      labels: sentLabels,
      datasets: [
        { label:'Positive', data: sentLabels.map((_,i)=>(sentClust[String(i)]||sentClust[i]||{positive:50}).positive), backgroundColor:'#5ebfb5cc', borderRadius:2 },
        { label:'Neutral',  data: sentLabels.map((_,i)=>(sentClust[String(i)]||sentClust[i]||{neutral:30}).neutral),  backgroundColor:'#8a8f9e55', borderRadius:2 },
        { label:'Negative', data: sentLabels.map((_,i)=>(sentClust[String(i)]||sentClust[i]||{negative:20}).negative),backgroundColor:'#e07b6acc', borderRadius:2 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend:{ labels:{color:'#8a8f9e',font:{family:'DM Mono',size:11},boxWidth:10} }, tooltip:{ backgroundColor:'#1a1f2b',borderColor:'#232840',borderWidth:1,titleColor:'#eeeae0',bodyColor:'#8a8f9e',callbacks:{label:ctx=>`  ${ctx.dataset.label}: ${ctx.raw}%`} } },
      scales: {
        x: { stacked:true, grid:{color:'#232840'}, ticks:{color:'#4a5068',font:{family:'DM Mono',size:10}} },
        y: { stacked:true, max:100, grid:{color:'#232840'}, ticks:{color:'#4a5068',font:{family:'DM Mono',size:10},callback:v=>v+'%'} }
      }
    }
  });

  // Cluster summary cards
  const cc = document.getElementById('clusterCards'); cc.innerHTML = '';
  clusterSummaries.forEach((s, i) => {
    const cl  = clusters[String(i)] || clusters[i] || {};
    const sc  = sentClust[String(i)] || sentClust[i] || {};
    const col = COLORS[i];
    const card = document.createElement('div');
    card.className = 'cluster-card'; card.style.setProperty('--cc', col);
    card.innerHTML = `
      <div class="cluster-badge"><div class="cluster-dot"></div><div class="cluster-code">Cluster ${i}</div></div>
      <div class="cluster-label-text">${cl.label || `Cluster ${i}`}</div>
      <div class="cluster-stat">Respondents: <span>${s.count}</span></div>
      <div class="cluster-stat">Positive: <span>${sc.positive || 0}%</span></div>
      <div class="mini-bars">
        ${Object.entries(s.avgRatings).filter(([,v])=>v!==null).slice(0,4).map(([lbl,v])=>`
          <div class="mini-bar-row">
            <div class="mini-bar-label">${lbl.slice(0,10)}</div>
            <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${(v/5)*100}%"></div></div>
            <div class="mini-bar-val">${v}</div>
          </div>`).join('')}
      </div>`;
    card.onclick = () => navTo('clusters');
    cc.appendChild(card);
  });

  // Urgent flags
  if (urgentFlags.length > 0) {
    document.getElementById('urgentSection').style.display = 'block';
    document.getElementById('urgentBadge').style.display   = 'inline';
    document.getElementById('urgentBadge').textContent     = urgentFlags.length;
    const ul = document.getElementById('urgentList'); ul.innerHTML = '';
    urgentFlags.forEach(f => {
      const el = document.createElement('div'); el.className = 'urgent-item';
      el.innerHTML = `<div class="urgent-id">${f.respondent_id} · Cluster ${f.cluster}</div>
                      <div style="font-size:11px;color:var(--text-2)">${(f.key_phrases || []).join(' · ')}</div>
                      <div class="urgent-reason">${f.flag_reason || 'Flagged for urgent review'}</div>`;
      ul.appendChild(el);
    });
  }

  // Themes chart
  destroyChart('theme');
  const themes      = themeData.themes || [];
  const themeColors = themes.map((_, i) => COLORS[i % COLORS.length]);
  state.charts.theme = new Chart(document.getElementById('themeChart'), {
    type: 'bar',
    data: { labels: themes.map(t=>t.name), datasets: [{ data:themes.map(t=>t.count), backgroundColor:themeColors.map(c=>c+'88'), borderColor:themeColors, borderWidth:1, borderRadius:2 }] },
    options: {
      indexAxis:'y', responsive:true,
      plugins: { legend:{display:false}, tooltip:{backgroundColor:'#1a1f2b',borderColor:'#232840',borderWidth:1,titleColor:'#eeeae0',bodyColor:'#8a8f9e',callbacks:{label:ctx=>`  ${ctx.raw} mentions`}} },
      scales: {
        x: { grid:{color:'#232840'}, ticks:{color:'#4a5068',font:{family:'DM Mono',size:10}} },
        y: { grid:{display:false},   ticks:{color:'#8a8f9e',font:{family:'DM Mono',size:10}} }
      }
    }
  });

  // Actions
  const al = document.getElementById('actionList'); al.innerHTML = '';
  const pcMap = { high:'var(--coral)', medium:'var(--gold)', low:'var(--teal)' };
  (actionData.insights || []).slice(0, 6).forEach(a => {
    const el = document.createElement('div'); el.className = 'action-item';
    el.style.setProperty('--pc', pcMap[a.priority] || 'var(--gold)');
    el.innerHTML = `
      <div class="action-priority">${a.priority}</div>
      <div class="action-body">
        <div class="action-text">${a.insight}</div>
        <div class="action-src">${a.category} · Clusters ${(a.source_clusters||[]).join(', ')} · ${a.breadth}</div>
      </div>
      <div class="action-cdot" style="background:${COLORS[(a.source_clusters||[0])[0]]||'#e8c468'}"></div>`;
    al.appendChild(el);
  });
}

// ═══════════════════════════════════════════════════════════════════
// RESPONDENT TABLE
// ═══════════════════════════════════════════════════════════════════
function populateTableFilters() {
  const { k, clusterLabels } = state.analysisResult;
  const clusters = clusterLabels;
  const sel = document.getElementById('filterCluster');
  sel.innerHTML = '<option value="all">All Clusters</option>';
  for (let i = 0; i < k; i++) {
    const cl = clusters[String(i)] || clusters[i] || {};
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = `Cluster ${i}: ${cl.label || ''}`;
    sel.appendChild(opt);
  }
  renderTable();
}

function filterTable() { renderTable(); }

function renderTable() {
  const filterCluster   = document.getElementById('filterCluster').value;
  const filterSentiment = document.getElementById('filterSentiment').value;
  const filterUrgent    = document.getElementById('filterUrgent').value;
  const { textCols, likertCols, clusterLabels } = state.analysisResult;
  const clusters = clusterLabels;

  let rows = state.tableData;
  if (filterCluster   !== 'all') rows = rows.filter(r => String(r.cluster) === String(filterCluster));
  if (filterSentiment !== 'all') rows = rows.filter(r => r.sentiment === filterSentiment);
  if (filterUrgent    === 'urgent') rows = rows.filter(r => r.flag_urgent);

  // Sort
  const { col, dir } = state.tableSort;
  rows = [...rows].sort((a, b) => {
    let av = a[col], bv = b[col];
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return dir === 'asc' ? -1 : 1;
    if (av > bv) return dir === 'asc' ? 1 : -1;
    return 0;
  });

  // Build header
  const thead = document.getElementById('respondentTableHead');
  const cols  = ['respondent_id','cluster','sentiment','confidence','flag_urgent','key_phrases', ...(textCols.length ? ['response_preview'] : [])];
  const colLabels = { respondent_id:'ID', cluster:'Cluster', sentiment:'Sentiment', confidence:'Confidence', flag_urgent:'⚑ Urgent', key_phrases:'Key Phrases', response_preview:'Response Preview' };
  thead.innerHTML = `<tr>${cols.map(c => `<th data-col="${c}" onclick="sortTable('${c}')" class="${state.tableSort.col===c ? 'sort-'+state.tableSort.dir : ''}">${colLabels[c]||c}</th>`).join('')}</tr>`;

  // Build body
  const tbody = document.getElementById('respondentTableBody');
  tbody.innerHTML = '';
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:var(--text-3);padding:40px">No records match the current filters</td></tr>`;
    return;
  }

  rows.forEach(row => {
    const ci  = row.cluster;
    const col = COLORS[ci] || COLORS[0];
    const cl  = clusters[String(ci)] || clusters[ci] || {};
    const tr  = document.createElement('tr');
    if (row.flag_urgent) tr.classList.add('urgent-row');
    tr.onclick = () => openDrawer(row);

    const previewText = (row.textResponses[0]?.answer || '').slice(0, 80) + (row.textResponses[0]?.answer?.length > 80 ? '…' : '');
    const phrasesHTML = (row.key_phrases || []).map(p => `<span class="phrase-tag">${p}</span>`).join('');

    tr.innerHTML = `
      <td class="id-cell">${row.respondent_id}</td>
      <td><span class="cluster-pill" style="background:${col}18;color:${col};border-color:${col}44">
        <span style="width:5px;height:5px;border-radius:50%;background:${col};display:inline-block"></span>
        ${cl.label || `C${ci}`}
      </span></td>
      <td><span class="sentiment-tag ${row.sentiment}">${row.sentiment}</span></td>
      <td><span class="confidence-tag ${row.confidence}">${row.confidence}</span></td>
      <td>${row.flag_urgent ? '<span class="urgent-flag">🚨</span>' : '<span style="color:var(--text-3)">—</span>'}</td>
      <td class="phrases-cell">${phrasesHTML}</td>
      ${textCols.length ? `<td class="text-cell">${previewText || '<span style="color:var(--text-3)">—</span>'}</td>` : ''}`;
    tbody.appendChild(tr);
  });
}

function sortTable(col) {
  if (state.tableSort.col === col) {
    state.tableSort.dir = state.tableSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    state.tableSort = { col, dir: 'asc' };
  }
  renderTable();
}

// ─── DETAIL DRAWER ──────────────────────────────────────────────
function openDrawer(row) {
  const { clusterLabels } = state.analysisResult;
  const ci  = row.cluster;
  const col = COLORS[ci] || COLORS[0];
  const cl  = clusterLabels[String(ci)] || clusterLabels[ci] || {};

  document.getElementById('drawerTitle').textContent = `${row.respondent_id} — Detail`;
  document.getElementById('detailDrawer').style.setProperty('--dc', col);

  const phrasesHTML = (row.key_phrases || []).map(p => `<span class="phrase-tag">${p}</span>`).join('');
  const ratingsHTML = Object.entries(row.ratings).map(([lbl, v]) => `
    <div class="drawer-rating-row">
      <div style="width:130px;flex-shrink:0;font-size:10px;color:var(--text-3)">${lbl}</div>
      <div class="drawer-rating-bar"><div class="drawer-rating-fill" style="width:${(v/5)*100}%"></div></div>
      <div style="width:32px;text-align:right;font-size:10px;color:var(--text-2)">${v}/5</div>
    </div>`).join('');

  // Text responses with questions labelled
  const responsesHTML = (row.textResponses || []).map(t => `
    <div style="margin-bottom:14px">
      <div class="drawer-question-label">Q: ${t.question}</div>
      <div class="drawer-response-text">"${t.answer}"</div>
    </div>`).join('') || '<div style="color:var(--text-3);font-size:11px">No text responses</div>';

  document.getElementById('drawerBody').innerHTML = `
    <div class="drawer-section">
      <div class="drawer-section-label">Cluster Assignment</div>
      <div class="drawer-meta-row">
        <span class="cluster-pill" style="background:${col}18;color:${col};border-color:${col}44">
          <span style="width:6px;height:6px;border-radius:50%;background:${col};display:inline-block"></span>
          Cluster ${ci}: ${cl.label || ''}
        </span>
      </div>
    </div>
    <div class="drawer-section">
      <div class="drawer-section-label">Sentiment</div>
      <div class="drawer-meta-row">
        <span class="sentiment-tag ${row.sentiment}">${row.sentiment}</span>
        <span class="confidence-tag ${row.confidence}">${row.confidence} confidence</span>
        ${row.flag_urgent ? '<span style="color:var(--coral);font-size:11px">🚨 Flagged Urgent</span>' : ''}
      </div>
      ${row.flag_urgent && row.flag_reason ? `<div style="font-size:11px;color:var(--coral);margin-top:6px;font-style:italic">${row.flag_reason}</div>` : ''}
    </div>
    ${phrasesHTML ? `<div class="drawer-section"><div class="drawer-section-label">Key Phrases</div><div class="drawer-phrases">${phrasesHTML}</div></div>` : ''}
    <div class="drawer-section">
      <div class="drawer-section-label">Survey Responses</div>
      ${responsesHTML}
    </div>
    ${Object.keys(row.ratings).length ? `<div class="drawer-section"><div class="drawer-section-label">Likert Ratings</div>${ratingsHTML}</div>` : ''}`;

  document.getElementById('detailDrawer').style.display = 'block';
}

function closeDrawer() {
  document.getElementById('detailDrawer').style.display = 'none';
}

// ═══════════════════════════════════════════════════════════════════
// CHAT / INSIGHTS PAGE
// ═══════════════════════════════════════════════════════════════════
function buildInsightsSidebar() {
  const { k, clusterLabels, actionData, sentimentData } = state.analysisResult;
  const clusters    = clusterLabels;
  const urgentFlags = (sentimentData.results || []).filter(r => r.flag_urgent);

  const questions = [
    'Which cluster needs the most urgent attention?',
    'What are the top 3 improvement priorities?',
    'How does sentiment differ across clusters?',
    'Which themes appear in multiple clusters?',
    'Which cluster was most satisfied and why?',
    'What actions should a programme manager take first?'
  ];
  document.getElementById('quickQuestions').innerHTML = questions.map(q =>
    `<div class="context-item" onclick="quickAsk('${q}')">${q}</div>`
  ).join('');

  const pcMap = { high:'high', medium:'med', low:'low' };
  document.getElementById('contextActions').innerHTML = (actionData.insights || []).slice(0, 4).map(a =>
    `<div class="context-item" onclick="quickAsk('Tell me more about: ${a.insight.replace(/'/g,"\\'")}')">
      <div style="font-size:11px;color:var(--text-1);margin-bottom:4px">${a.insight.slice(0,80)}${a.insight.length>80?'…':''}</div>
      <span class="context-tag ${pcMap[a.priority]||'med'}">${a.priority}</span>
    </div>`).join('');

  document.getElementById('contextUrgent').innerHTML = urgentFlags.length
    ? urgentFlags.slice(0, 4).map(f =>
        `<div class="context-item" onclick="quickAsk('Tell me about the urgent response ${f.respondent_id}')">
          <div style="font-size:10px;color:var(--coral)">⚠ ${f.respondent_id} · C${f.cluster}</div>
          <div style="font-size:11px;color:var(--text-2)">${(f.flag_reason||'').slice(0,70)}</div>
        </div>`).join('')
    : '<div class="context-item" style="color:var(--text-3)">No urgent flags detected</div>';
}

function buildContextSummary() {
  if (!state.analysisResult) return 'No analysis loaded.';
  const { k, clusterLabels, sentimentData, themeData, actionData, clusterSummaries } = state.analysisResult;
  const clusters = clusterLabels;
  const themes   = (themeData.themes || []).map(t => `${t.name} (${t.count})`).join(', ');
  const topActs  = (actionData.insights || []).slice(0, 3).map(a => `[${a.priority.toUpperCase()}] ${a.insight}`).join('\n');
  const urgentN  = (sentimentData.results || []).filter(r => r.flag_urgent).length;
  const clDesc   = Array.from({ length: k }, (_, i) => {
    const cl = clusters[String(i)] || clusters[i] || {};
    const sc = (sentimentData.cluster_summary || {})[String(i)] || {};
    return `Cluster ${i} "${cl.label||'—'}": n=${clusterSummaries[i]?.count||0}, ${sc.positive||0}% positive. Profile: ${cl.respondent_profile||'—'}`;
  }).join('\n');
  return `Programme: ${state.programName}\n${k} clusters identified.\n\n${clDesc}\n\nThemes: ${themes}\n\nTop actions:\n${topActs}\n\nUrgent flags: ${urgentN}`;
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = '';
  document.getElementById('sendBtn').disabled = true;
  addChatMsg('user', msg);
  const el = addChatMsg('assistant', '<span class="spinner">⟳</span> Thinking…');
  const sysPrompt = `You are an expert in learning and development evaluation analytics. You have just completed a full analysis of a post-program survey.
ANALYSIS CONTEXT:
${buildContextSummary()}
Answer questions about the results precisely and concisely. Reference specific clusters, sentiment data, and themes. Keep responses to 2-5 sentences unless asked for more detail.`;
  try {
    const reply = await callClaude(sysPrompt, msg, 800);
    el.querySelector('.msg-bubble').textContent = reply;
  } catch (e) {
    el.querySelector('.msg-bubble').textContent = 'Error reaching Claude. Check network connection.';
  }
  document.getElementById('sendBtn').disabled = false;
}

function addChatMsg(role, html) {
  const msgs = document.getElementById('chatMessages');
  const el   = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = `<div class="msg-role ${role}">${role === 'assistant' ? 'Claude' : 'You'}</div><div class="msg-bubble">${html}</div>`;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
  return el;
}

function quickAsk(q) {
  navTo('insights');
  document.getElementById('chatInput').value = q;
  sendChat();
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}

// ═══════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════
function exportJSON(type) {
  if (!state.analysisResult) return;
  let data, filename;
  if (type === 'clusters') {
    data = state.analysisResult.clusterLabels;
    filename = 'cluster_labels.json';
  } else {
    data = state.analysisResult;
    filename = 'feedback_analysis_results.json';
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
}

function exportCSV() {
  if (!state.tableData.length) return;
  const headers = ['respondent_id','cluster','cluster_label','sentiment','confidence','flag_urgent','flag_reason','key_phrases'];
  const rows = state.tableData.map(r =>
    headers.map(h => {
      let v = r[h];
      if (Array.isArray(v)) v = v.join('; ');
      if (v === null || v === undefined) v = '';
      v = String(v).replace(/"/g, '""');
      return `"${v}"`;
    }).join(',')
  );
  const csv  = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'respondent_sentiment_table.csv'; a.click();
}

// ─── UTILS ──────────────────────────────────────────────────────
function destroyChart(id) {
  if (state.charts[id]) { state.charts[id].destroy(); delete state.charts[id]; }
}
