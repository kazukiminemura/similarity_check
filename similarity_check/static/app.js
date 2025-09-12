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
  lab.textContent = `${label}: ${item.name || ''}`;

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

  document.getElementById('search-btn').onclick = async () => {
    const target = targetSel.value;
    const topk = Number(topkEl.value || 5);
    const frame_stride = Number(strideEl.value || 5);
    const device = (deviceSel?.value || 'auto');
    grid.innerHTML = 'Searching...';
    try {
      const res = await fetchJSON('/api/search_clips', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, device, topk, frame_stride }),
      });
      grid.innerHTML = '';
      const first = createVideoCell('Target', res.target, true);
      grid.appendChild(first.cell);
      for (let i = 0; i < 5; i++) {
        const item = res.results[i];
        const cell = document.createElement('div');
        cell.className = 'cell';
        if (!item) {
          cell.innerHTML = '<div class="label">(empty)</div>';
          grid.appendChild(cell);
          continue;
        }
        const { cell: c } = createVideoCell(`Top ${i+1}`, item, false);
        grid.appendChild(c);
      }
      setUpSync(grid);
    } catch (e) {
      grid.innerHTML = 'Error: ' + e;
    }
  };

  // Force recompute (ignore caches)
  const recomputeBtn = document.getElementById('recompute-btn');
  if (recomputeBtn) {
    recomputeBtn.onclick = () => {
      document.getElementById('search-btn').onclick = null; // avoid double bind
      (async () => {
        const target = targetSel.value;
        const topk = Number(topkEl.value || 5);
        const frame_stride = Number(strideEl.value || 5);
        const device = (deviceSel?.value || 'auto');
        grid.innerHTML = 'Recomputing...';
        try {
          const res = await fetchJSON('/api/search_clips', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target, device, topk, frame_stride, recompute: true }),
          });
          grid.innerHTML = '';
          const first = createVideoCell('Target', res.target, true);
          grid.appendChild(first.cell);
          for (let i = 0; i < 5; i++) {
            const item = res.results[i];
            const cell = document.createElement('div');
            cell.className = 'cell';
            if (!item) {
              cell.innerHTML = '<div class="label">(empty)</div>';
              grid.appendChild(cell);
              continue;
            }
            const { cell: c } = createVideoCell(`Top ${i+1}`, item, false);
            grid.appendChild(c);
          }
          setUpSync(grid);
        } catch (e) {
          grid.innerHTML = 'Error: ' + e;
        }
      })();
    };
  }

  try {
    const data = await fetchJSON('/api/clips');
    data.clips.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      targetSel.appendChild(opt);
    });
  } catch (e) {
    // ignore
  }
}

window.addEventListener('DOMContentLoaded', init);
