/**
 * Construction Equipment Monitor — app.js
 * Vanilla JS — no frameworks.
 *
 * Functions:
 *   checkHealth()
 *   setStatus(state)
 *   initSSE()
 *   updateEquipmentCards(equipment)
 *   buildCard(item)
 *   updateCard(card, item)
 *   updateDashboard(equipment)
 *   animateNumber(el, target, suffix, decimals)
 *   initUpload()
 *   initVideo()
 */

const API = 'http://localhost:8000';

// ── Stored dashboard state ─────────────────────────────────────────────────
const _prevStats = { active: 0, idle: 0, util: 0, count: 0 };

/* ──────────────────────────────────────────────────────────────────────────
   Health check
   ────────────────────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch(`/api/health`);
    const d = await r.json();
    setStatus(d.status === 'ok' ? 'online' : 'offline');
  } catch {
    setStatus('offline');
  }
}

function setStatus(state) {
  const pill = document.getElementById('status-pill');
  const text = document.getElementById('status-text');
  pill.className = 'status-pill';

  switch (state) {
    case 'online':
      pill.classList.add('status-online');
      text.textContent = 'Pipeline Running';
      break;
    case 'reconnecting':
      pill.classList.add('status-reconnecting');
      text.textContent = 'Reconnecting…';
      break;
    default:
      pill.classList.add('status-offline');
      text.textContent = 'Pipeline Offline';
  }
}

setInterval(checkHealth, 5000);
checkHealth();

/* ──────────────────────────────────────────────────────────────────────────
   SSE — live equipment stream
   ────────────────────────────────────────────────────────────────────────── */
function initSSE() {
  let es;

  function connect() {
    es = new EventSource(`${API}/stream`);

    es.onmessage = (e) => {
      try {
        const equipment = JSON.parse(e.data);
        updateEquipmentCards(equipment);
        updateDashboard(equipment);
      } catch {
        // malformed JSON — skip
      }
    };

    es.onerror = () => {
      setStatus('reconnecting');
      es.close();
      // auto-reconnect after 3 s
      setTimeout(connect, 3000);
    };
  }

  connect();
}

initSSE();

/* ──────────────────────────────────────────────────────────────────────────
   Equipment cards
   ────────────────────────────────────────────────────────────────────────── */

/** Return CSS badge class for an activity string. */
function activityBadgeClass(activity) {
  const map = {
    DIGGING: 'badge-digging',
    LOADING: 'badge-loading',
    DUMPING: 'badge-dumping',
    WAITING: 'badge-waiting',
  };
  return map[(activity || '').toUpperCase()] || 'badge-waiting';
}

/** Return fill class for a utilization percentage. */
function progressClass(pct) {
  if (pct >= 70) return 'fill-green';
  if (pct >= 40) return 'fill-amber';
  return 'fill-red';
}

/** Build a new equipment card DOM element. */
function buildCard(item) {
  const card = document.createElement('div');
  card.className = 'equip-card';
  card.dataset.id = item.equipment_id;

  card.innerHTML = `
    <div class="card-top">
      <span class="card-id">${item.equipment_id}</span>
      <span class="card-state">
        <span class="dot"></span>
        <span class="state-label"></span>
      </span>
    </div>
    <div class="card-class"></div>

    <hr class="card-divider" />

    <div class="card-row">
      <span class="card-label">Activity</span>
      <span class="activity-badge card-activity"></span>
    </div>
    <div class="card-row">
      <span class="card-label">Confidence</span>
      <span class="card-val card-confidence"></span>
    </div>

    <hr class="card-divider" />

    <div class="card-row">
      <span class="card-label">Working</span>
      <span class="card-val card-active"></span>
    </div>
    <div class="card-row">
      <span class="card-label">Idle</span>
      <span class="card-val card-idle"></span>
    </div>

    <hr class="card-divider" />

    <div class="progress-row">
      <div class="progress-track">
        <div class="progress-fill card-progress-fill"></div>
      </div>
      <span class="progress-pct card-progress-pct"></span>
    </div>
  `;

  updateCard(card, item, false);
  return card;
}

/** Update an existing card's values, optionally flashing it. */
function updateCard(card, item, flash = true) {
  const state = (item.current_state || 'INACTIVE').toUpperCase();
  const activity = (item.current_activity || 'WAITING').toUpperCase();
  const util = parseFloat(item.utilization_percent ?? item.utilization_pct ?? 0);
  const active = parseFloat(item.total_active_seconds ?? item.active_seconds ?? 0);
  const idle = parseFloat(item.total_idle_seconds ?? item.idle_seconds ?? 0);
  const conf = item.confidence != null ? Math.round(item.confidence * 100) + '%' : '—';

  // State class
  card.classList.remove('state-active', 'state-inactive');
  card.classList.add(state === 'ACTIVE' ? 'state-active' : 'state-inactive');

  // Text fields
  card.querySelector('.state-label').textContent = state;
  card.querySelector('.card-class').textContent = item.equipment_class || '—';

  const badge = card.querySelector('.card-activity');
  badge.textContent = activity;
  badge.className = `activity-badge card-activity ${activityBadgeClass(activity)}`;

  card.querySelector('.card-confidence').textContent = conf;
  card.querySelector('.card-active').textContent = active.toFixed(1) + 's';
  card.querySelector('.card-idle').textContent = idle.toFixed(1) + 's';

  // Progress bar
  const fill = card.querySelector('.card-progress-fill');
  fill.style.width = Math.min(util, 100) + '%';
  fill.className = `progress-fill card-progress-fill ${progressClass(util)}`;
  card.querySelector('.card-progress-pct').textContent = util.toFixed(1) + '%';

  // Flash animation on update
  if (flash) {
    card.classList.remove('flash');
    void card.offsetWidth; // reflow to restart animation
    card.classList.add('flash');
  }
}

/** Update all equipment cards from SSE data array. */
function updateEquipmentCards(equipment) {
  const container = document.getElementById('equipment-cards');
  const noMsg = document.getElementById('no-equipment-msg');
  const badge = document.getElementById('equip-count-badge');

  if (!Array.isArray(equipment) || equipment.length === 0) {
    if (noMsg) noMsg.style.display = 'block';
    badge.textContent = 0;
    Array.from(container.children).forEach(el => {
      if (el.id !== 'no-equipment-msg') el.remove();
    });
    return;
  }

  if (noMsg) noMsg.style.display = 'none';
  badge.textContent = equipment.length;

  equipment.forEach((item) => {
    const id = item.equipment_id;
    let card = container.querySelector(`[data-id="${CSS.escape(id)}"]`);

    if (!card) {
      card = buildCard(item);
      container.appendChild(card);
    } else {
      updateCard(card, item);
    }
  });
}

/* ──────────────────────────────────────────────────────────────────────────
   Dashboard stats
   ────────────────────────────────────────────────────────────────────────── */

/**
 * Animate a numeric display from its current value to a new target.
 * @param {HTMLElement} el
 * @param {number} target
 * @param {string} suffix  e.g. 's', '%', ''
 * @param {number} decimals
 */
function animateNumber(el, target, suffix = '', decimals = 1) {
  const current = parseFloat(el.dataset.raw ?? '0');
  if (current === target) return;

  const steps = 20;
  const delta = (target - current) / steps;
  let step = 0;

  const id = setInterval(() => {
    step++;
    const val = current + delta * step;
    el.textContent = (step >= steps ? target : val).toFixed(decimals) + suffix;
    if (step >= steps) {
      clearInterval(id);
      el.dataset.raw = target;
    }
  }, 50);
}

/** Recalculate and animate the 4 stat cards from the latest SSE snapshot. */
function updateDashboard(equipment) {
  if (!Array.isArray(equipment) || equipment.length === 0) {
    animateNumber(document.getElementById('stat-active'), 0, 's', 1);
    animateNumber(document.getElementById('stat-idle'), 0, 's', 1);
    animateNumber(document.getElementById('stat-util'), 0, '%', 1);
    animateNumber(document.getElementById('stat-count'), 0, '', 0);
    return;
  }

  const totalActive = equipment.reduce(
    (s, e) => s + parseFloat(e.total_active_seconds ?? e.active_seconds ?? 0), 0
  );
  const totalIdle = equipment.reduce(
    (s, e) => s + parseFloat(e.total_idle_seconds ?? e.idle_seconds ?? 0), 0
  );
  const avgUtil = equipment.reduce(
    (s, e) => s + parseFloat(e.utilization_percent ?? e.utilization_pct ?? 0), 0
  ) / equipment.length;
  const count = equipment.length;

  animateNumber(document.getElementById('stat-active'), totalActive, 's', 1);
  animateNumber(document.getElementById('stat-idle'),   totalIdle,   's', 1);
  animateNumber(document.getElementById('stat-util'),   avgUtil,     '%', 1);
  animateNumber(document.getElementById('stat-count'),  count,       '',  0);
}

/* ──────────────────────────────────────────────────────────────────────────
   Video feed
   ────────────────────────────────────────────────────────────────────────── */
function initVideo() {
  const img = document.getElementById('video-feed');
  const placeholder = document.getElementById('video-placeholder');
  const ts = document.getElementById('stream-timestamp');

  // Update timestamp every second while stream is alive
  let tsInterval = null;

  img.onload = () => {
    placeholder.style.display = 'none';
    img.style.display = 'block';
    if (!tsInterval) {
      tsInterval = setInterval(() => {
        const now = new Date();
        ts.textContent = now.toLocaleTimeString();
      }, 1000);
    }
  };

  img.onerror = () => {
    img.style.display = 'none';
    placeholder.style.display = 'flex';
    clearInterval(tsInterval);
    tsInterval = null;
    ts.textContent = '—';
    // retry every 3 s
    setTimeout(() => {
      img.src = `${API}/video/feed?_t=${Date.now()}`;
    }, 3000);
  };
}

initVideo();

/* ──────────────────────────────────────────────────────────────────────────
   Video upload
   ────────────────────────────────────────────────────────────────────────── */
function initUpload() {
  const dropZone  = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const btn       = document.getElementById('upload-btn');
  const btnText   = document.getElementById('upload-btn-text');
  const spinner   = document.getElementById('upload-spinner');
  const msgEl     = document.getElementById('upload-msg');
  const fileLabel = document.getElementById('drop-filename');

  let selectedFile = null;

  // ── File selection helpers ────────────────────────────────────────────────
  function formatBytes(bytes) {
    if (bytes < 1024)       return bytes + ' B';
    if (bytes < 1048576)    return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  function acceptFile(file) {
    if (!file) return;
    const ok = /\.(mp4|avi|mov)$/i.test(file.name);
    if (!ok) {
      showMsg('Only .mp4, .avi, .mov files are accepted.', 'error');
      return;
    }
    selectedFile = file;
    fileLabel.textContent = `${file.name}  (${formatBytes(file.size)})`;
    fileLabel.style.display = 'block';
    btn.disabled = false;
    clearMsg();
  }

  // ── Click to browse ───────────────────────────────────────────────────────
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) acceptFile(fileInput.files[0]);
  });

  // ── Drag-and-drop ─────────────────────────────────────────────────────────
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  ['dragleave', 'dragend'].forEach((ev) =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove('drag-over'))
  );

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    acceptFile(file);
  });

  // ── Upload ────────────────────────────────────────────────────────────────
  btn.addEventListener('click', async () => {
    if (!selectedFile) return;

    btn.disabled = true;
    btnText.textContent = 'Uploading…';
    spinner.style.display = 'inline-block';
    clearMsg();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const r = await fetch(`${API}/video/upload`, {
        method: 'POST',
        body: formData,
      });

      if (r.ok) {
        const d = await r.json();
        showMsg(`✓ Processing started (job ${d.job_id})`, 'success');
        selectedFile = null;
        fileLabel.style.display = 'none';
        fileInput.value = '';
        
        // Instantly force UI clear since database was truncated
        updateDashboard([]);
        updateEquipmentCards([]);
        document.getElementById('video-placeholder').style.display = 'flex';
      } else {
        const err = await r.text();
        showMsg(`Upload failed: ${err}`, 'error');
        btn.disabled = false;
      }
    } catch (err) {
      showMsg(`Upload failed: ${err.message}`, 'error');
      btn.disabled = false;
    } finally {
      btnText.textContent = 'Start Processing';
      spinner.style.display = 'none';
    }
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  function showMsg(text, type) {
    msgEl.textContent = text;
    msgEl.className = `upload-msg ${type}`;
    msgEl.style.display = 'block';
  }

  function clearMsg() {
    msgEl.style.display = 'none';
    msgEl.textContent = '';
  }
}

initUpload();
