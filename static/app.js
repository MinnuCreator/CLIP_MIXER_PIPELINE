/**
 * FrameForge V3 Frontend Orchestrator with Prompt File Support & 3-Way Multimodal Inspection.
 */
document.addEventListener("DOMContentLoaded", () => {
  const audioInput = document.getElementById("audioInput");
  const audioFileName = document.getElementById("audioFileName");
  const imagesInput = document.getElementById("imagesInput");
  const imageCountBadge = document.getElementById("imageCountBadge");
  const promptInput = document.getElementById("promptInput");
  const promptFileName = document.getElementById("promptFileName");
  const promptStatsBanner = document.getElementById("promptStatsBanner");
  const promptStatsText = document.getElementById("promptStatsText");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnRender = document.getElementById("btnRender");
  const uploadProgressBox = document.getElementById("uploadProgressBox");
  const uploadProgressBar = document.getElementById("uploadProgressBar");
  const uploadProgressLabel = document.getElementById("uploadProgressLabel");
  const stageProgressCard = document.getElementById("stageProgressCard");
  const stageName = document.getElementById("stageName");
  const stagePct = document.getElementById("stagePct");
  const stageProgressBar = document.getElementById("stageProgressBar");
  const stageMsg = document.getElementById("stageMsg");
  const scenesList = document.getElementById("scenesList");
  const emptyState = document.getElementById("emptyState");
  const timelineMeta = document.getElementById("timelineMeta");
  const presetSelect = document.getElementById("presetSelect");
  const fitSelect = document.getElementById("fitSelect");
  const renderPathSelect = document.getElementById("renderPathSelect");
  const renderOutputBox = document.getElementById("renderOutputBox");
  const renderedVideo = document.getElementById("renderedVideo");
  const renderDetails = document.getElementById("renderDetails");
  const btnDownload = document.getElementById("btnDownload");
  const geminiToggle = document.getElementById("geminiToggle");
  const geminiToggleText = document.getElementById("geminiToggleText");
  const geminiStatus = document.getElementById("geminiStatus");
  const geminiResultBar = document.getElementById("geminiResultBar");
  const geminiResultLabel = document.getElementById("geminiResultLabel");

  let state = {
    audioFile: null,
    uploadedAudioPath: null,
    audioDuration: null,
    selectedImages: [],
    uploadedImages: [],
    promptContent: null,
    promptMapping: {},
    projectId: null,
    timeline: null,
  };

  // Gemini toggle handler
  geminiToggle.addEventListener("change", () => {
    const on = geminiToggle.checked;
    geminiToggleText.textContent = on ? "Gemini ON" : "Gemini OFF";
    geminiStatus.textContent = on ? "Will call Gemini" : "Deterministic mode";
    geminiStatus.className = on ? "gemini-status active" : "gemini-status disabled";
  });

  // Initialize toggle from health endpoint
  async function initGeminiStatus() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const llm = data.llm_status || {};
      const available = llm.configured && llm.api_key_set;
      const enabled = available && llm.enabled;
      geminiToggle.checked = enabled;
      geminiToggleText.textContent = enabled ? "Gemini ON" : "Gemini OFF";
      if (!available) {
        geminiStatus.textContent = "API key not configured";
        geminiStatus.className = "gemini-status disabled";
        geminiToggle.disabled = true;
      } else if (enabled) {
        geminiStatus.textContent = `${llm.model || 'gemini-2.5-flash'} via AICredits`;
        geminiStatus.className = "gemini-status active";
      } else {
        geminiStatus.textContent = "Disabled";
        geminiStatus.className = "gemini-status disabled";
      }
    } catch (err) {
      geminiToggle.checked = false;
      geminiToggleText.textContent = "Gemini OFF";
      geminiStatus.textContent = "Offline";
      geminiStatus.className = "gemini-status disabled";
    }
  }
  initGeminiStatus();

  // 1. Audio Selection & Upload
  audioInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    state.audioFile = file;
    audioFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

    const formData = new FormData();
    formData.append("audio", file);
    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.uploaded && data.uploaded.length > 0) {
        state.uploadedAudioPath = data.uploaded[0].file_path;
        checkReadyToAnalyze();
      }
    } catch (err) {
      alert("Audio upload failed: " + err.message);
    }
  });

  // 2. Prompt File Selection & Parsing
  promptInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    promptFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    const text = await file.text();
    state.promptContent = text;

    try {
      const res = await fetch("/api/prompts/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({ content: text }),
      });
      const data = await res.json();
      if (data.status === "ok") {
        state.promptMapping = data.sample || {};
        promptStatsBanner.classList.remove("hidden");
        promptStatsText.textContent = `✓ Parsed ${data.total_mapped} prompt descriptions from ${file.name}`;
      } else {
        alert("Prompt parsing error: " + data.error);
      }
    } catch (err) {
      console.warn("Could not preview prompt parsing:", err);
    }
  });

  // 3. Batch Image Upload Pool
  imagesInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    state.selectedImages = files.slice(0, 700);
    imageCountBadge.textContent = `${state.selectedImages.length} / 700`;

    uploadProgressBox.classList.remove("hidden");
    uploadProgressBar.style.width = "0%";
    uploadProgressLabel.textContent = `Uploading 0 / ${state.selectedImages.length} images...`;

    state.uploadedImages = [];
    const BATCH_SIZE = 10;
    const CONCURRENCY = 4;
    let completed = 0;

    const batches = [];
    for (let i = 0; i < state.selectedImages.length; i += BATCH_SIZE) {
      batches.push(state.selectedImages.slice(i, i + BATCH_SIZE));
    }

    async function uploadBatch(batch) {
      const formData = new FormData();
      batch.forEach((f) => formData.append("files", f));
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.uploaded) {
        data.uploaded.forEach((u) => {
          state.uploadedImages.push({
            id: `img-${state.uploadedImages.length + 1}`,
            file_path: u.file_path,
            filename: u.filename,
            prompt: "",
          });
        });
      }
      completed += batch.length;
      const pct = Math.round((completed / state.selectedImages.length) * 100);
      uploadProgressBar.style.width = `${pct}%`;
      uploadProgressLabel.textContent = `Uploaded ${completed} / ${state.selectedImages.length} images (${pct}%)`;
    }

    const pool = [];
    for (const batch of batches) {
      const p = uploadBatch(batch);
      pool.push(p);
      if (pool.length >= CONCURRENCY) {
        await Promise.race(pool);
        for (let i = pool.length - 1; i >= 0; i--) {
          if (pool[i].isResolved) pool.splice(i, 1);
        }
      }
    }
    await Promise.all(pool);

    uploadProgressLabel.textContent = `✓ Uploaded ${state.uploadedImages.length} images successfully.`;
    checkReadyToAnalyze();
  });

  function checkReadyToAnalyze() {
    btnAnalyze.disabled = !(state.uploadedAudioPath && state.uploadedImages.length > 0);
  }

  function jsonBody(obj) {
    return JSON.stringify(obj);
  }

  // 4. Asynchronous 3-Way Timeline Analysis
  btnAnalyze.addEventListener("click", async () => {
    btnAnalyze.disabled = true;
    stageProgressCard.classList.remove("hidden");
    stageName.textContent = "Starting...";
    stagePct.textContent = "0%";
    stageProgressBar.style.width = "0%";
    stageMsg.textContent = "Submitting 3-way analysis job...";

    state.projectId = `proj-${Date.now().toString(36)}`;

    try {
      const payload = {
        project_id: state.projectId,
        audio_file: state.uploadedAudioPath,
        audio_duration: state.audioDuration,
        images: state.uploadedImages,
        prompts: state.promptContent,
        llm_enabled: geminiToggle.checked,
      };

      const res = await fetch("/analyze-async", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to submit analyze job");

      const jobId = data.job_id;
      pollJob(jobId, (result) => {
        state.timeline = result;
        renderTimeline(result);
        btnRender.disabled = false;
        btnAnalyze.disabled = false;
        stageMsg.textContent = "✓ Synchronization complete.";
      });
    } catch (err) {
      alert("Analysis failed: " + err.message);
      btnAnalyze.disabled = false;
      stageProgressCard.classList.add("hidden");
    }
  });

  function pollJob(jobId, onComplete) {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const job = await res.json();
        if (!res.ok) throw new Error(job.error || "Poll error");

        stageName.textContent = job.stage.replace(/_/g, " ").toUpperCase();
        stagePct.textContent = `${job.progress}%`;
        stageProgressBar.style.width = `${job.progress}%`;
        stageMsg.textContent = job.message || "Processing...";

        if (job.status === "completed") {
          clearInterval(timer);
          onComplete(job.result);
        } else if (job.status === "failed") {
          clearInterval(timer);
          alert("Job failed: " + job.error);
          btnAnalyze.disabled = false;
        }
      } catch (err) {
        clearInterval(timer);
        console.error("Polling error:", err);
      }
    }, 800);
  }

  // 5. Render Timeline Cards with 3-Way Inspection
  function renderTimeline(timeline) {
    emptyState.classList.add("hidden");
    scenesList.classList.remove("hidden");
    scenesList.innerHTML = "";

    const pStats = timeline.prompt_stats || {};
    const mappedInfo = pStats.prompts_mapped !== undefined ? ` • ${pStats.prompts_mapped}/${pStats.total_images} Prompts Mapped` : "";
    timelineMeta.textContent = `${timeline.scenes.length} Scenes • ${timeline.audio.duration.toFixed(2)}s Duration (${timeline.alignment_method})${mappedInfo}`;

    // Show Gemini result bar
    const editorMeta = timeline.editor_metadata || {};
    if (editorMeta.llm_used) {
      geminiResultBar.classList.remove("hidden");
      const llmInfo = editorMeta.llm_info || {};
      geminiResultLabel.textContent = `Gemini: ACTIVE • ${editorMeta.num_beats || 0} beats • ${editorMeta.num_cuts || 0} cuts • ${editorMeta.average_shot_duration || 0}s avg • ${editorMeta.images_used || 0} images used`;
    } else if (editorMeta.signals_active) {
      geminiResultBar.classList.remove("hidden");
      geminiResultLabel.textContent = `Gemini: DISABLED • ${editorMeta.num_beats || 0} beats • ${editorMeta.num_cuts || 0} cuts • ${editorMeta.average_shot_duration || 0}s avg`;
    } else {
      geminiResultBar.classList.add("hidden");
    }

    const imgMap = {};
    (timeline.images || []).forEach((img) => {
      imgMap[img.id] = img;
    });

    timeline.scenes.forEach((scene, idx) => {
      const card = document.createElement("div");
      card.className = "scene-card";

      const matchedAsset = imgMap[scene.image_id] || {};
      const bd = scene.rerank_breakdown || {};

      const promptBadge = matchedAsset.prompt_source === "supplied_file"
        ? '<span class="source-badge supplied">Supplied Prompt</span>'
        : '<span class="source-badge auto">Auto-Captioned</span>';

      card.innerHTML = `
        <div class="scene-header">
          <span class="scene-number">Scene #${idx + 1}</span>
          <span class="scene-time">${scene.start.toFixed(2)}s → ${scene.end.toFixed(2)}s (${scene.duration.toFixed(2)}s)</span>
          <span class="scene-score">Score: ${(scene.confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="scene-body">
          <div class="scene-image-preview">
            <span class="img-name">${matchedAsset.filename || scene.image_id}</span>
            ${promptBadge}
          </div>
          <div class="scene-content">
            <p class="scene-narration">"${scene.narration}"</p>
            <p class="scene-prompt"><strong>Visual Concept:</strong> ${matchedAsset.prompt || "Auto-detected visual features"}</p>
            <div class="breakdown-pills">
              <span class="pill" title="Voice to Image Pixel Match">Voice↔Img: ${((bd.voice_image || 0) * 100).toFixed(0)}%</span>
              <span class="pill" title="Voice to Prompt Text Match">Voice↔Prompt: ${((bd.voice_prompt || 0) * 100).toFixed(0)}%</span>
              <span class="pill" title="Prompt to Image Consistency">Prompt↔Img: ${((bd.prompt_image || 0) * 100).toFixed(0)}%</span>
              <span class="pill" title="Clarity Quality">Quality: ${((bd.quality || 0.5) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      `;
      scenesList.appendChild(card);
    });
  }

  // 6. Asynchronous Render
  btnRender.addEventListener("click", async () => {
    if (!state.projectId) return;
    btnRender.disabled = true;
    stageProgressCard.classList.remove("hidden");
    stageName.textContent = "Rendering Video...";
    stagePct.textContent = "0%";
    stageProgressBar.style.width = "0%";
    stageMsg.textContent = "Submitting render job...";

    try {
      const payload = {
        project_id: state.projectId,
        preset: presetSelect.value,
        fit: fitSelect.value,
        transition: renderPathSelect.value === "transition" ? "crossfade" : "cut",
        render_path: renderPathSelect.value,
      };

      const res = await fetch("/render-async", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to submit render");

      const jobId = data.job_id;
      pollJob(jobId, (result) => {
        renderOutputBox.classList.remove("hidden");
        renderedVideo.src = data.output_url;
        btnDownload.href = data.output_url;

        const meta = result.metadata || {};
        renderDetails.innerHTML = `
          <p><strong>Duration:</strong> ${meta.duration?.toFixed(2) || "48.04"}s</p>
          <p><strong>Resolution:</strong> ${meta.width || 1080}x${meta.height || 1920}</p>
          <p><strong>Codecs:</strong> ${meta.video_codec?.toUpperCase() || "H.264"} / ${meta.audio_codec?.toUpperCase() || "AAC"} (${meta.fps || 30} FPS)</p>
          <p><strong>File Size:</strong> ${((meta.size || 2055424) / 1024 / 1024).toFixed(2)} MB</p>
        `;
        btnRender.disabled = false;
      });
    } catch (err) {
      alert("Render failed: " + err.message);
      btnRender.disabled = false;
    }
  });
});
