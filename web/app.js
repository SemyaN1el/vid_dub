const state = {
  mode: "path",
  selectedJobId: "",
  pollTimer: null,
  toastTimer: null,
};

const el = {
  healthPill: document.querySelector("#healthPill"),
  healthText: document.querySelector("#healthText"),
  modeButtons: document.querySelectorAll(".mode-button"),
  pathField: document.querySelector(".path-field"),
  uploadField: document.querySelector(".upload-field"),
  jobForm: document.querySelector("#jobForm"),
  startButton: document.querySelector("#startButton"),
  refreshButton: document.querySelector("#refreshButton"),
  artifactsButton: document.querySelector("#artifactsButton"),
  copyLogButton: document.querySelector("#copyLogButton"),
  jobSelect: document.querySelector("#jobSelect"),
  jobStatus: document.querySelector("#jobStatus"),
  jobNameView: document.querySelector("#jobNameView"),
  returnCode: document.querySelector("#returnCode"),
  steps: document.querySelectorAll(".step"),
  artifactList: document.querySelector("#artifactList"),
  logOutput: document.querySelector("#logOutput"),
  toast: document.querySelector("#toast"),
};

function iconize() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.add("visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    el.toast.classList.remove("visible");
  }, 3600);
}

function requestJson(url, options = {}) {
  return fetch(url, options).then(async (response) => {
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) {
      const detail = data?.detail || response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  });
}

function setMode(mode) {
  state.mode = mode;
  el.modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  el.pathField.classList.toggle("hidden", mode !== "path");
  el.uploadField.classList.toggle("hidden", mode !== "upload");
}

function readOptions() {
  return {
    step: document.querySelector("#step").value,
    job_name: document.querySelector("#jobName").value.trim() || null,
    test: document.querySelector("#testMode").checked,
    resume: document.querySelector("#resume").checked,
    skip_metrics: document.querySelector("#skipMetrics").checked,
    mt_provider: document.querySelector("#mtProvider").value,
    mt_model: document.querySelector("#mtModel").value.trim() || null,
    mt_strategy: document.querySelector("#mtStrategy").value,
    mt_style: document.querySelector("#mtStyle").value,
    tts_provider: document.querySelector("#ttsProvider").value,
    subtitle_mode: document.querySelector("#subtitleMode").value,
  };
}

async function checkHealth() {
  try {
    await requestJson("/health");
    el.healthPill.classList.remove("fail");
    el.healthPill.classList.add("ok");
    el.healthText.textContent = "online";
  } catch {
    el.healthPill.classList.remove("ok");
    el.healthPill.classList.add("fail");
    el.healthText.textContent = "offline";
  }
}

async function refreshJobs() {
  const data = await requestJson("/jobs");
  el.jobSelect.innerHTML = "";

  if (!data.jobs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No jobs";
    el.jobSelect.append(option);
    clearJobView();
    return;
  }

  for (const job of data.jobs) {
    const option = document.createElement("option");
    option.value = job.id;
    option.textContent = `${job.output_job_name} · ${job.status}`;
    el.jobSelect.append(option);
  }

  if (!state.selectedJobId || !data.jobs.some((job) => job.id === state.selectedJobId)) {
    state.selectedJobId = data.jobs[0].id;
  }
  el.jobSelect.value = state.selectedJobId;
  await loadJob(state.selectedJobId);
}

function clearJobView() {
  state.selectedJobId = "";
  el.jobStatus.textContent = "idle";
  el.jobNameView.textContent = "none";
  el.returnCode.textContent = "-";
  el.logOutput.textContent = "Выбери или запусти задачу, чтобы увидеть лог.";
  el.artifactList.innerHTML = '<p class="empty-state">Готовые файлы появятся после завершения пайплайна.</p>';
  paintSteps(null, "");
}

function paintSteps(job, logText) {
  const stepNames = Array.from(el.steps).map((step) => step.dataset.step);
  let activeIndex = -1;
  const lowerLog = (logText || "").toLowerCase();
  const metricsSkipped = Boolean(job?.command?.includes("--skip-metrics"));

  if (job?.status === "succeeded") {
    activeIndex = stepNames.length;
  } else {
    activeIndex = stepNames.findIndex((name) => lowerLog.includes(`шаг`) && lowerLog.includes(name));
    if (activeIndex < 0 && job?.status === "running") {
      activeIndex = 0;
    }
  }

  el.steps.forEach((step, index) => {
    step.classList.remove("active", "done", "skipped");
    if (step.dataset.step === "metrics" && metricsSkipped) {
      step.classList.add("skipped");
      return;
    }
    if (job?.status === "succeeded" || index < activeIndex) {
      step.classList.add("done");
    } else if (index === activeIndex && job?.status === "running") {
      step.classList.add("active");
    }
  });
}

async function loadJob(jobId) {
  if (!jobId) {
    clearJobView();
    return;
  }

  const [job, logData, artifactData] = await Promise.all([
    requestJson(`/jobs/${jobId}`),
    requestJson(`/jobs/${jobId}/logs`).catch(() => ({ log: "" })),
    requestJson(`/jobs/${jobId}/artifacts`).catch(() => ({ artifacts: [] })),
  ]);

  state.selectedJobId = job.id;
  el.jobStatus.textContent = job.status;
  el.jobNameView.textContent = job.output_job_name;
  el.returnCode.textContent = job.return_code ?? "-";
  el.logOutput.textContent = logData.log || "Лог пока пуст.";
  paintSteps(job, logData.log || "");
  renderArtifacts(artifactData.artifacts || []);

  const shouldPoll = job.status === "queued" || job.status === "running";
  schedulePolling(shouldPoll);
}

function renderArtifacts(artifacts) {
  el.artifactList.innerHTML = "";
  if (!artifacts.length) {
    el.artifactList.innerHTML = '<p class="empty-state">Готовые файлы появятся после завершения пайплайна.</p>';
    return;
  }

  for (const artifact of artifacts) {
    const item = document.createElement("div");
    item.className = "artifact-item";
    item.innerHTML = `
      <div>
        <div class="artifact-name">${escapeHtml(artifact.path)}</div>
        <div class="artifact-meta">${formatBytes(artifact.size_bytes)}</div>
      </div>
      <div class="artifact-actions">
        <a href="${artifact.download_url}" title="Open or download" target="_blank" rel="noreferrer">
          <i data-lucide="download"></i>
        </a>
      </div>
    `;
    el.artifactList.append(item);
  }
  iconize();
}

function schedulePolling(enabled) {
  window.clearInterval(state.pollTimer);
  state.pollTimer = null;
  if (!enabled || !state.selectedJobId) {
    return;
  }
  state.pollTimer = window.setInterval(() => {
    loadJob(state.selectedJobId).catch((error) => showToast(error.message));
  }, 3500);
}

async function createPathJob() {
  const videoPath = document.querySelector("#videoPath").value.trim();
  if (!videoPath) {
    throw new Error("Укажи путь к видео внутри контейнера.");
  }
  return requestJson("/jobs/from-path", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_path: videoPath,
      options: readOptions(),
    }),
  });
}

async function createUploadJob() {
  const file = document.querySelector("#videoFile").files[0];
  if (!file) {
    throw new Error("Выбери видеофайл для загрузки.");
  }

  const options = readOptions();
  const form = new FormData();
  form.append("video", file);
  form.append("step", options.step);
  form.append("job_name", options.job_name || "");
  form.append("test", String(options.test));
  form.append("resume", String(options.resume));
  form.append("skip_metrics", String(options.skip_metrics));
  form.append("mt_provider", options.mt_provider);
  form.append("mt_model", options.mt_model || "");
  form.append("mt_strategy", options.mt_strategy);
  form.append("mt_style", options.mt_style);
  form.append("tts_provider", options.tts_provider);
  form.append("subtitle_mode", options.subtitle_mode);

  return requestJson("/jobs", {
    method: "POST",
    body: form,
  });
}

async function submitJob(event) {
  event.preventDefault();
  el.startButton.disabled = true;
  try {
    const job = state.mode === "path" ? await createPathJob() : await createUploadJob();
    state.selectedJobId = job.id;
    showToast(`Задача ${job.output_job_name} запущена.`);
    await refreshJobs();
  } catch (error) {
    showToast(error.message);
  } finally {
    el.startButton.disabled = false;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvents() {
  el.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  el.jobForm.addEventListener("submit", submitJob);
  el.refreshButton.addEventListener("click", () => refreshJobs().catch((error) => showToast(error.message)));
  el.artifactsButton.addEventListener("click", () => {
    if (state.selectedJobId) {
      loadJob(state.selectedJobId).catch((error) => showToast(error.message));
    }
  });
  el.jobSelect.addEventListener("change", () => {
    state.selectedJobId = el.jobSelect.value;
    loadJob(state.selectedJobId).catch((error) => showToast(error.message));
  });
  el.copyLogButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(el.logOutput.textContent);
    showToast("Лог скопирован.");
  });
}

async function init() {
  bindEvents();
  iconize();
  setMode("path");
  await checkHealth();
  await refreshJobs().catch((error) => showToast(error.message));
}

init().catch((error) => showToast(error.message));
