async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function createVideoCell(label, item, isMaster=false) {
  const cell = document.createElement('div');
  cell.className = 'cell';
  const lab = document.createElement('div');
  lab.className = 'label';
  const displayName = item?.display_name || item?.name || '';
  lab.textContent = `${label}: ${displayName}`;

  const url = item.clip_url || item.url; // play from /clips when available
  const useClip = !!item.clip_url;

  const vid = document.createElement('video');
  vid.src = url;
  vid.controls = true;
  vid.preload = 'metadata';
  vid.playsInline = true;
  if (isMaster) vid.dataset.master = '1';

  if (!useClip) {
    if (item.start != null) vid.dataset.clipStart = String(item.start);
    if (item.end != null) vid.dataset.clipEnd = String(item.end);
    vid.addEventListener('loadedmetadata', () => {
      const s = parseFloat(vid.dataset.clipStart || 'NaN');
      if (!Number.isNaN(s)) { try { vid.currentTime = s; } catch(_){} }
    });
    vid.addEventListener('timeupdate', () => {
      const s = parseFloat(vid.dataset.clipStart || 'NaN');
      const e = parseFloat(vid.dataset.clipEnd || 'NaN');
      if (!Number.isNaN(s) && vid.currentTime < s - 0.05) {
        try { vid.currentTime = s; } catch(_){}
      }
      if (!Number.isNaN(e) && vid.currentTime > e - 0.05) {
        try { vid.pause(); vid.currentTime = e; } catch(_){}
      }
    });
  }

  cell.appendChild(lab);
  cell.appendChild(vid);
  return { cell, vid };
}

function setUpSync(gridEl) {
  const syncToggle = document.getElementById('sync-toggle');
  const playAll = document.getElementById('play-all');
  const pauseAll = document.getElementById('pause-all');

  function getVideos() {
    return Array.from(gridEl.querySelectorAll('video'));
  }
  function master() {
    return gridEl.querySelector('video[data-master="1"]');
  }

  playAll.onclick = () => {
    const vids = getVideos();
    vids.forEach(v => { try { v.play(); } catch(e){} });
  };
  pauseAll.onclick = () => {
    const vids = getVideos();
    vids.forEach(v => { try { v.pause(); } catch(e){} });
  };

  function syncToMaster() {
    const m = master();
    if (!m) return;
    const t = m.currentTime;
    const vids = getVideos().filter(v => v !== m);
    vids.forEach(v => {
      if (!syncToggle.checked) return;
      const diff = Math.abs(v.currentTime - t);
      if (diff > 0.12) {
        try { v.currentTime = t; } catch(e){}
      }
      if (!m.paused && v.paused) {
        try { v.play(); } catch(e){}
      }
      if (m.paused && !v.paused) {
        try { v.pause(); } catch(e){}
      }
    });
  }

  const m = master();
  if (m) m.addEventListener('timeupdate', syncToMaster);
}

async function init() {
  const targetSel = document.getElementById('target-select');
  const grid = document.getElementById('grid');
  const topkEl = document.getElementById('topk');
  const strideEl = document.getElementById('stride');
  const deviceSel = document.getElementById('device-select');
  const swingOnlyEl = document.getElementById('swing-only');
  const swingSecsEl = document.getElementById('swing-seconds');

  const renderResults = (res) => {
    grid.innerHTML = '';
    if (!res) {
      grid.innerHTML = '(no results)';
      return;
    }

    const clipPool = Array.isArray(res.clip_pool) ? res.clip_pool.slice() : [];
    const takeClip = () => (clipPool.length ? clipPool.shift() : null);
    const slotRaw = Number(topkEl.value || 5);
    const slotCount = Number.isFinite(slotRaw) && slotRaw > 0 ? slotRaw : 5;

    if (res.target) {
      const targetCell = createVideoCell('Target', res.target, true);
      grid.appendChild(targetCell.cell);
    } else {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.innerHTML = '<div class="label">(no target)</div>';
      grid.appendChild(cell);
    }

    for (let i = 0; i < slotCount; i++) {
      let item = Array.isArray(res.results) ? res.results[i] : null;
      if (!item) {
        item = takeClip();
      } else if (!item.clip_url) {
        const fallback = takeClip();
        if (fallback) {
          item = {
            ...item,
            clip_url: fallback.clip_url || fallback.url,
            clip_abs: fallback.clip_abs,
            clip_path: fallback.clip_path || fallback.path,
            clip_name: fallback.clip_name || fallback.name,
            display_name: fallback.display_name || item.display_name || item.name,
          };
        }
      }
      if (!item) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.innerHTML = '<div class="label">(empty)</div>';
        grid.appendChild(cell);
        continue;
      }
      const { cell: c } = createVideoCell(`Top ${i + 1}`, item, false);
      grid.appendChild(c);
    }

    setUpSync(grid);
  };

  const runSearch = async (overrides = {}) => {
    const target = targetSel.value;
    const topk = Number(topkEl.value || 5);
    const frame_stride = Number(strideEl.value || 5);
    const device = (deviceSel?.value || 'auto');
    const swing_only = !!(swingOnlyEl?.checked);
    const swing_seconds = Number(swingSecsEl?.value || 2.5);
    grid.innerHTML = overrides.recompute ? 'Recomputing...' : 'Searching...';
    try {
      const res = await fetchJSON('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, device, topk, frame_stride, swing_only, swing_seconds, ...overrides }),
      });
      renderResults(res);
    } catch (e) {
      grid.innerHTML = 'Error: ' + e;
    }
  };

  document.getElementById('search-btn').onclick = () => { runSearch(); };

  const recomputeBtn = document.getElementById('recompute-btn');
  if (recomputeBtn) {
    recomputeBtn.onclick = () => { runSearch({ recompute: true }); };
  }

  try {
    const data = await fetchJSON('/api/videos');
    targetSel.innerHTML = '';
    const videos = Array.isArray(data?.videos) ? data.videos : [];
    videos.forEach(entry => {
      const opt = document.createElement('option');
      if (entry && typeof entry === 'object') {
        const name = entry.name ?? entry.path ?? '';
        opt.value = name;
        opt.textContent = name;
      } else {
        const name = String(entry ?? '');
        opt.value = name;
        opt.textContent = name;
      }
      targetSel.appendChild(opt);
    });
    if (targetSel.options.length > 0) {
      targetSel.selectedIndex = 0;
    }
  } catch (e) {
    // ignore
  }
}

window.addEventListener('DOMContentLoaded', init);
