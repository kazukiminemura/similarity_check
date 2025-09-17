const API = {
  videos: "/api/videos",
  search: "/api/search"
};

const DEFAULTS = {
  topk: 5,
  swingSeconds: 5,
  frameStride: 5,
  swingOnly: true
};

const state = {
  videoMap: new Map(),
  loadingTargets: false,
  runningSearch: false
};

const els = {};

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return {};
}

function setStatus(line, detail) {
  if (els.statusLine) els.statusLine.textContent = line;
  if (els.statusDetail) els.statusDetail.textContent = detail;
}

function setSearchDisabled(disabled) {
  [els.searchBtn, els.recomputeBtn].forEach((button) => {
    if (button) button.disabled = disabled;
  });
}

function updateTargetPath(name) {
  if (!els.targetMeta) return;
  const entry = state.videoMap.get(name || "");
  if (!entry) {
    els.targetMeta.textContent = "";
    return;
  }
  if (entry.path) {
    els.targetMeta.textContent = entry.path;
    return;
  }
  const root = window.APP_CONFIG?.targetRoot;
  if (!root) {
    els.targetMeta.textContent = entry.name;
    return;
  }
  const separator = root.endsWith("/") || root.endsWith("\\") ? "" : "/";
  els.targetMeta.textContent = `${root}${separator}${entry.name}`;
}

function showPlaceholder(message) {
  if (!els.resultsList) return;
  els.resultsList.innerHTML = "";
  const item = document.createElement("li");
  item.className = "results__placeholder";
  item.textContent = message;
  els.resultsList.appendChild(item);
  if (els.resultsMeta) els.resultsMeta.textContent = "";
}

function createVideoElement(entry) {
  const src = entry?.clip_url || entry?.url;
  if (!src) return null;
  const video = document.createElement("video");
  video.className = "results__video";
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = src;

  const start = typeof entry?.start === "number" ? entry.start : undefined;
  const end = typeof entry?.end === "number" ? entry.end : undefined;

  if (start !== undefined || end !== undefined) {
    const startTime = Number(start);
    const endTime = Number(end);

    if (!Number.isNaN(startTime)) {
      video.addEventListener("loadedmetadata", () => {
        try {
          video.currentTime = startTime;
        } catch (error) {
          // ignore seek errors
        }
      }, { once: true });
    }

    video.addEventListener("timeupdate", () => {
      if (!Number.isNaN(startTime) && video.currentTime < startTime - 0.05) {
        try {
          video.currentTime = startTime;
        } catch (error) {
          // ignore seek errors
        }
      }
      if (!Number.isNaN(endTime) && video.currentTime > endTime - 0.05) {
        try {
          video.pause();
          video.currentTime = endTime;
        } catch (error) {
          // ignore seek errors
        }
      }
    });
  }

  return video;
}

function appendResultItem(label, entry, { fromClipPool = false } = {}) {
  if (!els.resultsList) return;
  const item = document.createElement("li");
  item.className = "results__item";

  const topLine = document.createElement("div");
  topLine.className = "results__topline";

  const title = document.createElement("strong");
  title.textContent = label;
  topLine.appendChild(title);

  if (typeof entry?.score === "number" && Number.isFinite(entry.score)) {
    const score = document.createElement("span");
    score.className = "results__badge";
    score.textContent = `score ${entry.score.toFixed(3)}`;
    topLine.appendChild(score);
  }

  if (fromClipPool) {
    const badge = document.createElement("span");
    badge.className = "results__badge";
    badge.textContent = "clip pool";
    topLine.appendChild(badge);
  }

  item.appendChild(topLine);

  if (!entry) {
    const empty = document.createElement("div");
    empty.className = "text-muted";
    empty.textContent = "(empty)";
    item.appendChild(empty);
    els.resultsList.appendChild(item);
    return;
  }

  const name = document.createElement("div");
  name.className = "results__name";
  name.textContent = entry.display_name || entry.name || entry.clip_name || "Unknown";
  item.appendChild(name);

  const video = createVideoElement(entry);
  if (video) {
    item.appendChild(video);
  }

  const pathText = entry.clip_path || entry.path;
  if (pathText) {
    const path = document.createElement("div");
    path.className = "text-muted";
    path.textContent = pathText;
    item.appendChild(path);
  }

  els.resultsList.appendChild(item);
}

function renderResults(payload, topk) {
  if (!els.resultsList) return;

  if (!payload) {
    showPlaceholder("No response from server.");
    return;
  }

  els.resultsList.innerHTML = "";

  const target = payload.target;
  if (target) {
    appendResultItem("Target", target);
  }

  const slots = Number.isFinite(topk) && topk > 0 ? topk : DEFAULTS.topk;
  const ranked = Array.isArray(payload.results) ? payload.results : [];
  const clipPool = Array.isArray(payload.clip_pool) ? [...payload.clip_pool] : [];

  let rankedCount = 0;
  let clipCount = 0;
  let emptyCount = 0;

  for (let index = 0; index < slots; index += 1) {
    let entry = ranked[index];
    let fromClipPool = false;

    if (!entry && clipPool.length) {
      entry = clipPool.shift();
      fromClipPool = true;
    }

    appendResultItem(`Top ${index + 1}`, entry, { fromClipPool });

    if (!entry) {
      emptyCount += 1;
    } else if (fromClipPool) {
      clipCount += 1;
    } else {
      rankedCount += 1;
    }
  }

  if (els.resultsMeta) {
    const device = payload.used_device || els.deviceSelect?.value || "AUTO";
    els.resultsMeta.textContent = `${rankedCount} ranked · ${clipCount} from clips · ${emptyCount} empty · device ${device}`;
  }
}

async function loadTargets() {
  if (state.loadingTargets || !els.targetSelect) return;
  state.loadingTargets = true;
  if (els.refreshTargets) els.refreshTargets.disabled = true;

  try {
    const data = await fetchJSON(API.videos);
    const videos = Array.isArray(data?.videos) ? data.videos : [];

    state.videoMap.clear();
    els.targetSelect.innerHTML = "";

    let firstName = "";
    for (const entry of videos) {
      if (!entry || typeof entry !== "object") continue;
      const baseName = entry.name || entry.display_name || (entry.path ? entry.path.split(/[\\/]/).pop() : "");
      if (!baseName) continue;
      entry.name = baseName;
      state.videoMap.set(baseName, entry);
      const option = document.createElement("option");
      option.value = baseName;
      option.textContent = baseName;
      els.targetSelect.appendChild(option);
      if (!firstName) firstName = baseName;
    }

    if (firstName) {
      els.targetSelect.value = firstName;
      updateTargetPath(firstName);
    } else {
      updateTargetPath("");
    }
  } catch (error) {
    const message = error?.message || "Failed to load targets";
    setStatus("Error", message);
    throw error;
  } finally {
    state.loadingTargets = false;
    if (els.refreshTargets) els.refreshTargets.disabled = false;
  }
}

function readNumber(input, fallback, minimum) {
  if (!input) return fallback;
  const value = Number(input.value);
  if (!Number.isFinite(value)) return fallback;
  if (typeof minimum === "number" && value < minimum) return fallback;
  return value;
}

async function runSearch({ recompute = false } = {}) {
  if (state.runningSearch) return;
  if (!els.targetSelect?.value) {
    setStatus("Error", "Select a target video first.");
    return;
  }

  state.runningSearch = true;
  setSearchDisabled(true);

  const target = els.targetSelect.value;
  const device = els.deviceSelect?.value || "AUTO";
  const topk = readNumber(els.topk, DEFAULTS.topk, 1);
  const swingSeconds = readNumber(els.swingSeconds, DEFAULTS.swingSeconds, 0.5);

  const pendingText = recompute ? "Recomputing features..." : "Searching...";
  setStatus("Working", pendingText);
  showPlaceholder(pendingText);

  try {
    const requestBody = {
      target,
      device,
      topk,
      swing_seconds: swingSeconds,
      frame_stride: DEFAULTS.frameStride,
      swing_only: DEFAULTS.swingOnly
    };
    if (recompute) requestBody.recompute = true;

    const data = await fetchJSON(API.search, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody)
    });

    renderResults(data, topk);
    const deviceUsed = data?.used_device || device;
    setStatus("Done", `Search finished on ${deviceUsed}.`);
  } catch (error) {
    const message = error?.message || "Search failed";
    setStatus("Error", message);
    showPlaceholder(message);
  } finally {
    state.runningSearch = false;
    setSearchDisabled(false);
  }
}

function bindEvents() {
  document.getElementById("search-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });

  els.recomputeBtn?.addEventListener("click", () => runSearch({ recompute: true }));

  els.refreshTargets?.addEventListener("click", async () => {
    setStatus("Working", "Reloading target list...");
    try {
      await loadTargets();
      setStatus("Idle", "Targets updated. Run a search when ready.");
    } catch (error) {
      // loadTargets already reported the error via setStatus
    }
  });

  els.targetSelect?.addEventListener("change", (event) => {
    updateTargetPath(event.target.value);
  });
}

async function init() {
  els.targetSelect = document.getElementById("target-select");
  els.refreshTargets = document.getElementById("refresh-targets");
  els.targetMeta = document.getElementById("target-path");
  els.deviceSelect = document.getElementById("device-select");
  els.topk = document.getElementById("topk");
  els.swingSeconds = document.getElementById("swing-seconds");
  els.searchBtn = document.getElementById("search-btn");
  els.recomputeBtn = document.getElementById("recompute-btn");
  els.statusLine = document.getElementById("status-line");
  els.statusDetail = document.getElementById("status-detail");
  els.resultsList = document.getElementById("results-list");
  els.resultsMeta = document.getElementById("results-meta");

  setStatus("Idle", "Select a target video and run a search.");
  showPlaceholder("Search results will appear here.");

  bindEvents();

  try {
    await loadTargets();
    if (els.targetSelect?.value) {
      setStatus("Idle", "Ready to search.");
    }
  } catch (error) {
    // loadTargets already surfaced the error message
  }
}

window.addEventListener("DOMContentLoaded", init);
