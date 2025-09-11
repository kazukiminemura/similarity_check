async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function createVideoCell(label, url, name, isMaster=false) {
  const cell = document.createElement('div');
  cell.className = 'cell';
  const lab = document.createElement('div');
  lab.className = 'label';
  lab.textContent = `${label}: ${name}`;
  const vid = document.createElement('video');
  vid.src = url;
  vid.controls = true;
  vid.preload = 'metadata';
  vid.playsInline = true;
  if (isMaster) vid.dataset.master = '1';
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
      if (diff > 0.12) { // allow small drift
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
  const candRoot = document.getElementById('cand-root');
  const grid = document.getElementById('grid');
  const topkEl = document.getElementById('topk');
  const strideEl = document.getElementById('stride');
  document.getElementById('search-btn').onclick = async () => {
    const target = targetSel.value;
    const topk = Number(topkEl.value || 5);
    const frame_stride = Number(strideEl.value || 5);
    grid.innerHTML = '検索中...';
    try {
      const res = await fetchJSON('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, candidates_dir: '.', topk, frame_stride }),
      });
      // Build 2x3 grid: target top-left + top5
      grid.innerHTML = '';
      const first = createVideoCell('Target', res.target.url, res.target.name, true);
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
        const { cell: c, vid } = createVideoCell(`Top ${i+1}`, item.url, item.name);
        grid.appendChild(c);
      }
      setUpSync(grid);
    } catch (e) {
      grid.innerHTML = 'エラー: ' + e;
    }
  };

  // Populate select
  try {
    const data = await fetchJSON('/api/videos');
    candRoot.textContent = data.root;
    data.videos.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      targetSel.appendChild(opt);
    });
  } catch (e) {
    candRoot.textContent = 'データフォルダが見つかりません';
  }
}

window.addEventListener('DOMContentLoaded', init);

