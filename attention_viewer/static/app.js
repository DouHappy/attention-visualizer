const state = {
  sampleId: 0,
  layer: 0,
  head: 0,
  layerCount: 0,
  headCount: 0,
  tokens: [],
  attention: [],
};

const sampleSlider = document.getElementById('sample-slider');
const sampleInput = document.getElementById('sample-input');
const layerSlider = document.getElementById('layer-slider');
const layerInput = document.getElementById('layer-input');
const headSlider = document.getElementById('head-slider');
const headInput = document.getElementById('head-input');
const sourceText = document.getElementById('source-text');
const predictionText = document.getElementById('prediction-text');
const fileInfo = document.getElementById('file-info');
const tokensContainer = document.getElementById('tokens-container');
const rowTokenLabel = document.getElementById('row-token');
const colTokenLabel = document.getElementById('col-token');
const attentionScoreLabel = document.getElementById('attention-score');

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function syncSliderAndInput(slider, input, callback) {
  slider.addEventListener('input', (event) => {
    input.value = event.target.value;
    callback(Number(event.target.value));
  });

  input.addEventListener('change', (event) => {
    const value = Number(event.target.value);
    slider.value = clamp(value, Number(slider.min), Number(slider.max));
    callback(Number(slider.value));
  });
}

function clearHighlights() {
  document.querySelectorAll('.token').forEach((node) => {
    node.classList.remove('highlight-row', 'highlight-col');
  });
  rowTokenLabel.textContent = '';
  colTokenLabel.textContent = '';
  attentionScoreLabel.textContent = '';
}

function renderTokens(tokens) {
  tokensContainer.innerHTML = '';
  tokens.forEach((token, index) => {
    const span = document.createElement('span');
    span.className = 'token';
    span.dataset.index = index;
    span.textContent = token;
    tokensContainer.appendChild(span);
  });
}

function highlightTokens(rowIndex, colIndex, value) {
  clearHighlights();
  const rowToken = tokensContainer.querySelector(`.token[data-index="${rowIndex}"]`);
  const colToken = tokensContainer.querySelector(`.token[data-index="${colIndex}"]`);
  if (rowToken) {
    rowToken.classList.add('highlight-row');
    rowTokenLabel.textContent = `${rowIndex}: ${rowToken.textContent}`;
  }
  if (colToken) {
    colToken.classList.add('highlight-col');
    colTokenLabel.textContent = `${colIndex}: ${colToken.textContent}`;
  }
  attentionScoreLabel.textContent = value.toFixed(6);
}

function renderHeatmap(tokens, attention) {
  const hoverText = attention.map((row, rowIndex) =>
    row.map((value, colIndex) => {
      const rowToken = tokens[rowIndex] ?? '';
      const colToken = tokens[colIndex] ?? '';
      return `行: ${rowIndex} ${rowToken}<br>列: ${colIndex} ${colToken}<br>Score: ${value.toFixed(6)}`;
    })
  );

  const data = [
    {
      z: attention,
      x: tokens.map((token, idx) => `${idx}`),
      y: tokens.map((token, idx) => `${idx}`),
      text: hoverText,
      type: 'heatmap',
      hoverinfo: 'text',
      colorscale: 'Viridis',
    },
  ];

  const layout = {
    xaxis: { title: 'Token index' },
    yaxis: { title: 'Token index' },
    margin: { t: 40, r: 0, l: 60, b: 60 },
  };

  Plotly.react('attention-plot', data, layout, { responsive: true });

  const plot = document.getElementById('attention-plot');
  if (plot && typeof plot.removeAllListeners === 'function') {
    plot.removeAllListeners('plotly_hover');
    plot.removeAllListeners('plotly_unhover');
  }
  plot.on('plotly_hover', (eventData) => {
    const point = eventData.points[0];
    highlightTokens(point.y, point.x, point.z);
  });

  plot.on('plotly_unhover', () => {
    clearHighlights();
  });

  clearHighlights();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

async function loadFilesMetadata() {
  const payload = await fetchJson('/api/files');
  const { sample_id_range: range } = payload;
  const minId = range?.min ?? 0;
  const maxId = range?.max ?? 0;
  sampleSlider.min = minId;
  sampleSlider.max = Math.max(minId, maxId);
  sampleSlider.value = minId;
  sampleInput.min = minId;
  sampleInput.max = Math.max(minId, maxId);
  sampleInput.value = minId;
  state.sampleId = minId;
  return payload;
}

async function loadAttention() {
  const params = new URLSearchParams({
    sample_id: state.sampleId,
    layer: state.layer,
    head: state.head,
  });
  let payload;
  try {
    payload = await fetchJson(`/api/attention?${params.toString()}`);
  } catch (error) {
    console.error('Failed to load attention payload', error);
    alert(`无法加载 attention 数据: ${error.message}`);
    return;
  }
  state.layerCount = payload.layer_count;
  state.headCount = payload.head_count;
  state.tokens = payload.tokens;
  state.attention = payload.attention;

  layerSlider.max = Math.max(0, payload.layer_count - 1);
  layerInput.max = Math.max(0, payload.layer_count - 1);
  headSlider.max = Math.max(0, payload.head_count - 1);
  headInput.max = Math.max(0, payload.head_count - 1);

  layerSlider.value = clamp(state.layer, Number(layerSlider.min), Number(layerSlider.max));
  layerInput.value = layerSlider.value;
  headSlider.value = clamp(state.head, Number(headSlider.min), Number(headSlider.max));
  headInput.value = headSlider.value;
  state.layer = Number(layerSlider.value);
  state.head = Number(headSlider.value);

  sourceText.textContent = payload.source || '';
  predictionText.textContent = payload.prediction || '';
  fileInfo.textContent = `${payload.file?.name ?? ''} | start: ${payload.file?.start_id ?? ''} | end: ${payload.file?.end_id ?? ''} | batch: ${payload.file?.batch_index ?? ''}`;

  renderTokens(payload.tokens);
  renderHeatmap(payload.tokens, payload.attention);
}

async function initialise() {
  try {
    await loadFilesMetadata();
    await loadAttention();
  } catch (error) {
    console.error('Failed to initialise application', error);
    alert(`初始化失败: ${error.message}`);
  }
}

syncSliderAndInput(sampleSlider, sampleInput, (value) => {
  state.sampleId = value;
  loadAttention();
});

syncSliderAndInput(layerSlider, layerInput, (value) => {
  state.layer = clamp(value, 0, Number(layerSlider.max));
  loadAttention();
});

syncSliderAndInput(headSlider, headInput, (value) => {
  state.head = clamp(value, 0, Number(headSlider.max));
  loadAttention();
});

initialise();
