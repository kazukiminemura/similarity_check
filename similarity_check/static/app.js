async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function createVideoCell(label, url, name, isMaster=false, startSec=null, endSec=null, debugLines=[]) {
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
  if (startSec != null) vid.dataset.clipStart = String(startSec);
  if (endSec != null) vid.dataset.clipEnd = String(endSec);
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
  cell.appendChild(lab);
  cell.appendChild(vid);
  if (debugLines && debugLines.length) {
    const dbg = document.createElement('small');
    dbg.style.display = 'block';
    dbg.style.opacity = '0.8';
    dbg.style.wordBreak = 'break-all';
    dbg.textContent = debugLines.join(' | ');
    cell.appendChild(dbg);
  }
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

  document.getElementById('search-btn').onclick = async () => {
    const target = targetSel.value;
    const topk = Number(topkEl.value || 5);
    const frame_stride = Number(strideEl.value || 5);
    const device = (deviceSel?.value || 'auto');
    const swing_only = !!(swingOnlyEl?.checked);
    const swing_seconds = Number(swingSecsEl?.value || 2.5);
    grid.innerHTML = 'Searching...';
    try {
      const res = await fetchJSON('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, device, topk, frame_stride, swing_only, swing_seconds }),
      });
      grid.innerHTML = '';
      const firstUrl = res.target.clip_url || res.target.url;
      const useClipForTarget = !!res.target.clip_url;
      const firstDbg = [];
      if (res.target.path_abs) firstDbg.push('orig=' + res.target.path_abs);
      if (res.target.clip_abs) firstDbg.push('clip=' + res.target.clip_abs);
      const first = createVideoCell(
        'Target',
        firstUrl,
        res.target.name,
        true,
        useClipForTarget ? null : res.target.start,
        useClipForTarget ? null : res.target.end,
        firstDbg
      );
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
        const url = item.clip_url || item.url;
        const useClip = !!item.clip_url;
        const dbg = [];
        if (item.path_abs) dbg.push('orig=' + item.path_abs);
        if (item.clip_abs) dbg.push('clip=' + item.clip_abs);
        const { cell: c } = createVideoCell(
          `Top ${i+1}`,
          url,
          item.name,
          false,
          useClip ? null : item.start,
          useClip ? null : item.end,
          dbg
        );
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
    recomputeBtn.onclick = async () => {
      const target = targetSel.value;
      const topk = Number(topkEl.value || 5);
      const frame_stride = Number(strideEl.value || 5);
      const device = (deviceSel?.value || 'auto');
      const swing_only = !!(swingOnlyEl?.checked);
      const swing_seconds = Number(swingSecsEl?.value || 2.5);
      grid.innerHTML = 'Recomputing...';
      try {
        const res = await fetchJSON('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target, device, topk, frame_stride, swing_only, swing_seconds, recompute: true }),
        });
        grid.innerHTML = '';
        const firstUrl = res.target.clip_url || res.target.url;
        const useClipForTarget = !!res.target.clip_url;
        const first = createVideoCell(
          'Target',
          firstUrl,
          res.target.name,
          true,
          useClipForTarget ? null : res.target.start,
          useClipForTarget ? null : res.target.end
        );
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
          const url = item.clip_url || item.url;
          const useClip = !!item.clip_url;
          const { cell: c } = createVideoCell(
            `Top ${i+1}`,
            url,
            item.name,
            false,
            useClip ? null : item.start,
            useClip ? null : item.end
          );
          grid.appendChild(c);
        }
        setUpSync(grid);
      } catch (e) {
        grid.innerHTML = 'Error: ' + e;
      }
    };
  }

  try {
    const data = await fetchJSON('/api/videos');
    data.videos.forEach(name => {
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
