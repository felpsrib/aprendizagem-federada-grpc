(() => {
  "use strict";

  const state = {
    snapshot: null,
    rpcFilter: "all",
    weightsRoundIndex: null,
    packets: [],
    packetSeq: 0,
  };

  const STATUS_LABELS = {
    idle: { text: "Ocioso", cls: "pill-neutral" },
    starting: { text: "Iniciando...", cls: "pill-blue" },
    running: { text: "Executando gRPC", cls: "pill-blue" },
    waiting_step: { text: "Aguardando Passo", cls: "pill-amber" },
    completed: { text: "Concluído", cls: "pill-emerald" },
    stopped: { text: "Parado", cls: "pill-neutral" },
    error: { text: "Erro", cls: "pill-red" },
  };

  const CLIENT_STATUS = {
    idle: { label: "Ocioso", color: "#94a3b8", pill: "pill-neutral" },
    polling: { label: "Consultando", color: "#2563eb", pill: "pill-blue" },
    training: { label: "Treinando", color: "#d97706", pill: "pill-amber" },
    sending: { label: "Enviando", color: "#059669", pill: "pill-emerald" },
    waiting: { label: "Aguardando", color: "#64748b", pill: "pill-neutral" },
    done: { label: "Concluído", color: "#059669", pill: "pill-emerald" },
    error: { label: "Erro", color: "#dc2626", pill: "pill-red" },
  };

  const STAGE_CAPTIONS = {
    0: "Etapa 0: Servidor inicializa o modelo global (pesos zerados)",
    1: "Etapa 1: Clientes consultam os parâmetros globais via GetGlobalModel",
    2: "Etapa 2: Clientes treinam LogisticRegression apenas com seus dados locais",
    3: "Etapa 3: Clientes serializam em Protobuf e enviam via SubmitUpdate",
    4: "Etapa 4: Servidor agrega parâmetros com FedAvg ponderado e avalia",
    5: "Etapa 5: Todas as rodadas concluídas — modelo global pronto",
  };

  const PRESETS = {
    setosa: { pl: 1.4, pw: 0.2, sl: 5.1, sw: 3.5 },
    versicolor: { pl: 4.5, pw: 1.5, sl: 6.0, sw: 2.8 },
    virginica: { pl: 5.8, pw: 2.2, sl: 6.8, sw: 3.1 },
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ------------------------------------------------------------------
  // Toast helper
  // ------------------------------------------------------------------
  let toastTimer = null;
  function showToast(msg, type = "info") {
    const el = $("#toast-banner");
    el.textContent = msg;
    el.className = `toast ${type}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 5000);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function fmtPct(v) {
    return v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
  }

  function fmtNum(v, digits = 4) {
    return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
  }

  // ------------------------------------------------------------------
  // Tabs
  // ------------------------------------------------------------------
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      $$(".tab").forEach((b) => {
        const active = b.dataset.tab === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      $$(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.id === `tab-${target}`);
      });
      if (target === "inference") {
        runPredict();
      }
    });
  });

  // ------------------------------------------------------------------
  // API Helpers
  // ------------------------------------------------------------------
  async function apiPost(url, body = {}) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `Erro HTTP ${resp.status}`);
    }
    return data;
  }

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------
  ["#cfg-clients", "#cfg-rounds"].forEach((sel) => {
    $(sel).addEventListener("input", () => {
      if (!state.snapshot || state.snapshot.status === "idle") {
        renderTopology();
        const r = Math.max(1, Math.min(25, Number($("#cfg-rounds").value) || 10));
        $("#kpi-round").textContent = `0 / ${r}`;
      }
    });
  });

  $("#btn-start").addEventListener("click", async () => {
    const rawDelay = Number($("#cfg-delay").value);
    const payload = {
      num_clients: Math.max(2, Math.min(10, Number($("#cfg-clients").value) || 5)),
      total_rounds: Math.max(1, Math.min(25, Number($("#cfg-rounds").value) || 10)),
      local_epochs: Math.max(1, Math.min(10, Number($("#cfg-epochs").value) || 1)),
      visual_delay: Number.isFinite(rawDelay) ? Math.max(0, Math.min(3, rawDelay)) : 0.4,
      step_mode: $("#cfg-step-mode").checked,
      poll_interval: 0.15,
      port: 50051,
    };
    $("#btn-start").disabled = true;
    try {
      const snap = await apiPost("/api/session/start", payload);
      state.weightsRoundIndex = null;
      applySnapshot(snap);
      showToast(`Sessão gRPC iniciada em ${snap.server_address} com ${payload.num_clients} clientes.`, "success");
    } catch (err) {
      showToast(`Falha ao iniciar sessão: ${err.message}`, "error");
    } finally {
      $("#btn-start").disabled = false;
    }
  });

  $("#btn-next-round").addEventListener("click", async () => {
    try {
      const snap = await apiPost("/api/session/next-round");
      applySnapshot(snap);
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  $("#btn-stop").addEventListener("click", async () => {
    try {
      const snap = await apiPost("/api/session/stop");
      applySnapshot(snap);
      showToast("Sessão gRPC encerrada.", "info");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  $("#btn-baseline").addEventListener("click", async () => {
    const btn = $("#btn-baseline");
    btn.disabled = true;
    btn.textContent = "Executando baseline...";
    try {
      const res = await apiPost("/api/manual-baseline", {
        num_clients: Number($("#cfg-clients").value) || 5,
        total_rounds: Number($("#cfg-rounds").value) || 10,
        local_epochs: Number($("#cfg-epochs").value) || 1,
      });
      showToast(`Baseline manual concluído em ${res.elapsed_ms} ms.`, "success");
    } catch (err) {
      showToast(`Erro no baseline manual: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Comparar Baseline Manual";
    }
  });

  // Fault injection buttons
  $$("[data-inject]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const kind = btn.dataset.inject;
      const box = $("#inject-result");
      try {
        const res = await apiPost("/api/session/inject-rpc", { kind, cid: 0 });
        box.classList.remove("hidden");
        box.innerHTML = `
          <strong>Resposta gRPC (UpdateAck)</strong> —
          accepted=<span class="${res.accepted ? "accent-emerald" : ""}" style="color:${res.accepted ? "#059669" : "#dc2626"}">${res.accepted}</span> |
          server_round=${res.server_round} |
          message="${escapeHtml(res.message)}"
        `;
        showToast(`RPC de validação enviado (${kind}): ${res.message}`, res.accepted ? "info" : "success");
      } catch (err) {
        box.classList.remove("hidden");
        box.textContent = `Erro: ${err.message}`;
        showToast(err.message, "error");
      }
    });
  });

  // RPC Filter chips
  $$("#rpc-filters .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.rpcFilter = chip.dataset.filter;
      $$("#rpc-filters .chip").forEach((c) => c.classList.toggle("active", c === chip));
      renderRpcTable();
    });
  });

  // Weights round scrubber
  $("#weights-round-slider").addEventListener("input", (e) => {
    state.weightsRoundIndex = Number(e.target.value);
    renderWeightsHeatmap();
  });

  // Inference sliders & presets
  const sliders = [
    { id: "in-pl", out: "val-pl" },
    { id: "in-pw", out: "val-pw" },
    { id: "in-sl", out: "val-sl" },
    { id: "in-sw", out: "val-sw" },
  ];

  let predictTimer = null;
  function schedulePredict() {
    sliders.forEach(({ id, out }) => {
      $(`#${out}`).textContent = `${Number($(`#${id}`).value).toFixed(1)} cm`;
    });
    clearTimeout(predictTimer);
    predictTimer = setTimeout(runPredict, 80);
  }

  sliders.forEach(({ id }) => {
    $(`#${id}`).addEventListener("input", schedulePredict);
  });

  $$("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = PRESETS[btn.dataset.preset];
      $("#in-pl").value = p.pl;
      $("#in-pw").value = p.pw;
      $("#in-sl").value = p.sl;
      $("#in-sw").value = p.sw;
      schedulePredict();
    });
  });

  async function runPredict() {
    try {
      const res = await apiPost("/api/predict", {
        petal_length: Number($("#in-pl").value),
        petal_width: Number($("#in-pw").value),
        sepal_length: Number($("#in-sl").value),
        sepal_width: Number($("#in-sw").value),
      });
      $("#predict-class").textContent = res.predicted_class;
      $("#predict-confidence").textContent = `Confiança: ${(res.confidence * 100).toFixed(1)}%`;
      $("#predict-model-note").textContent =
        res.model_round === 0
          ? "Modelo global na Rodada 0 (pesos zerados antes do treino federado)"
          : `Modelo global agregado na Rodada ${res.model_round}`;

      const colors = ["#2563eb", "#059669", "#d97706"];
      $("#predict-bars").innerHTML = res.probabilities
        .map(
          (item, idx) => `
          <div>
            <div class="prob-row-top">
              <span>${escapeHtml(item.species)}</span>
              <span class="mono">${(item.probability * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-track">
              <div class="prob-fill" style="width:${(item.probability * 100).toFixed(1)}%; background:${colors[idx % colors.length]}"></div>
            </div>
          </div>`
        )
        .join("");
    } catch (err) {
      $("#predict-model-note").textContent = `Erro ao prever: ${err.message}`;
    }
  }

  // Unit tests runner
  $("#btn-run-tests").addEventListener("click", async () => {
    const btn = $("#btn-run-tests");
    btn.disabled = true;
    btn.textContent = "Executando testes...";
    try {
      const res = await apiPost("/api/tests/run");
      renderTestResults(res);
      showToast(
        res.success
          ? `Todos os ${res.passed_cases}/${res.total_cases} testes unitários passaram!`
          : `Falha em testes (${res.passed_cases}/${res.total_cases}).`,
        res.success ? "success" : "error"
      );
    } catch (err) {
      showToast(`Erro ao rodar testes: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Executar Todos os Testes";
    }
  });

  // ------------------------------------------------------------------
  // SSE Connection
  // ------------------------------------------------------------------
  function connectSSE() {
    const es = new EventSource("/api/events");
    es.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "state") {
          applySnapshot(msg.data);
        } else if (msg.type === "client_update" && state.snapshot) {
          const c = state.snapshot.clients.find((x) => x.cid === msg.data.cid);
          if (c) Object.assign(c, msg.data);
          renderTopology();
          renderClientsGrid();
        } else if (msg.type === "rpc" && state.snapshot) {
          const { entry, stats, received_cids } = msg.data;
          state.snapshot.rpc_stats = stats;
          state.snapshot.received_cids = received_cids;
          state.snapshot.rpc_logs.push(entry);
          if (state.snapshot.rpc_logs.length > 200) {
            state.snapshot.rpc_logs = state.snapshot.rpc_logs.slice(-200);
          }
          spawnPacket(entry);
          renderKpis();
          renderTopology();
          renderRpcTable();
        } else if (msg.type === "tests") {
          renderTestResults(msg.data);
        }
      } catch (err) {
        console.error("Erro processando SSE:", err);
      }
    };
    es.onerror = () => {
      // Browser reconecta automaticamente
    };
  }

  // ------------------------------------------------------------------
  // State Application & Renderers
  // ------------------------------------------------------------------
  function applySnapshot(snap) {
    const prevRoundsLen = state.snapshot ? state.snapshot.rounds_history.length : 0;
    state.snapshot = snap;

    const st = STATUS_LABELS[snap.status] || STATUS_LABELS.idle;
    const badge = $("#status-badge");
    badge.textContent = st.text;
    badge.className = `pill ${st.cls}`;

    const ep = $("#endpoint-badge");
    ep.textContent = snap.server_address ? `gRPC: ${snap.server_address}` : "gRPC: inativo";
    ep.className = `pill mono ${snap.server_address && snap.status !== "stopped" ? "pill-blue" : "pill-neutral"}`;

    const isActive = ["running", "starting", "waiting_step"].includes(snap.status);
    $("#btn-stop").disabled = !isActive && snap.status !== "completed";

    const nextBtn = $("#btn-next-round");
    if (snap.waiting_for_step && snap.status === "waiting_step") {
      nextBtn.classList.remove("hidden");
      nextBtn.textContent = `Avançar para Rodada ${snap.current_round}`;
    } else {
      nextBtn.classList.add("hidden");
    }

    renderStages();
    renderKpis();
    renderTopology();
    renderCharts();
    renderRoundsTable();
    renderClientsGrid();
    renderWeightsHeatmap();
    renderConfusionMatrix();
    renderRpcTable();

    if (snap.last_test_results) {
      renderTestResults(snap.last_test_results);
    }

    if (snap.rounds_history.length !== prevRoundsLen && $("#tab-inference").classList.contains("active")) {
      runPredict();
    }
  }

  function renderStages() {
    const snap = state.snapshot;
    if (!snap) return;
    const active = snap.active_stage ?? 0;
    $("#stage-caption").textContent = STAGE_CAPTIONS[active] || "";
    $$("#stages-list li").forEach((li) => {
      const s = Number(li.dataset.stage);
      li.classList.toggle("active", snap.status !== "idle" && s === active);
      li.classList.toggle("done", snap.status === "completed" || (snap.status !== "idle" && s < active));
    });
  }

  function renderKpis() {
    const snap = state.snapshot;
    if (!snap) return;

    const total = snap.config.total_rounds || 1;
    const completedRounds = Math.max(0, snap.rounds_history.length - 1);
    const pct = Math.min(100, Math.round((completedRounds / total) * 100));

    $("#kpi-round").textContent = `${completedRounds} / ${total}`;
    $("#kpi-round-bar").style.width = `${pct}%`;
    $("#kpi-elapsed").textContent = snap.elapsed_sec != null ? `Tempo decorrido: ${snap.elapsed_sec.toFixed(1)}s` : "Tempo: —";

    const latest = snap.rounds_history[snap.rounds_history.length - 1];
    if (latest) {
      $("#kpi-acc").textContent = fmtPct(latest.global_acc);
      $("#kpi-acc-sub").textContent = `Treino médio local: ${fmtPct(latest.avg_train_acc)}`;
      $("#kpi-loss").textContent = fmtNum(latest.global_loss, 4);
      $("#kpi-loss-sub").textContent = `Treino médio local: ${fmtNum(latest.avg_train_loss, 4)}`;
    } else {
      $("#kpi-acc").textContent = "—";
      $("#kpi-acc-sub").textContent = "Treino médio local: —";
      $("#kpi-loss").textContent = "—";
      $("#kpi-loss-sub").textContent = "Treino médio local: —";
    }

    const stats = snap.rpc_stats || {};
    $("#kpi-rpcs").textContent = `${stats.total_rpcs || 0} RPCs`;
    $("#kpi-rpcs-sub").textContent = `${formatBytes(stats.total_bytes || 0)} • ${stats.submit_accepted || 0} aceitos • ${stats.submit_rejected || 0} recusados`;
  }

  // ------------------------------------------------------------------
  // Topology SVG + Packet Animations
  // ------------------------------------------------------------------
  function getNodePositions(numClients) {
    const cx = 320;
    const cy = 210;
    const rx = 225;
    const ry = 148;
    const nodes = [];
    for (let i = 0; i < numClients; i++) {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / numClients;
      nodes.push({
        cid: i,
        x: cx + rx * Math.cos(angle),
        y: cy + ry * Math.sin(angle),
      });
    }
    return { cx, cy, nodes };
  }

  function spawnPacket(entry) {
    const snap = state.snapshot;
    if (!snap || !snap.clients.length) return;
    const n = snap.clients.length;
    if (entry.cid < 0 || entry.cid >= n) return;
    const { cx, cy, nodes } = getNodePositions(n);
    const node = nodes[entry.cid];
    const fromServer = entry.direction === "server->client";
    const color = !entry.accepted ? "#dc2626" : fromServer ? "#2563eb" : "#059669";

    state.packets.push({
      id: ++state.packetSeq,
      x1: fromServer ? cx : node.x,
      y1: fromServer ? cy : node.y,
      x2: fromServer ? node.x : cx,
      y2: fromServer ? node.y : cy,
      color,
      born: performance.now(),
      duration: 550,
    });
  }

  function renderTopology() {
    const svg = $("#topology-svg");
    const snap = state.snapshot;
    const numClients = snap && snap.clients.length ? snap.clients.length : Number($("#cfg-clients").value) || 5;
    const { cx, cy, nodes } = getNodePositions(numClients);

    const receivedSet = new Set(snap?.received_cids || []);
    const receivedCount = receivedSet.size;
    const ringRadius = 52;
    const circumference = 2 * Math.PI * ringRadius;
    const progressRatio = numClients > 0 ? receivedCount / numClients : 0;
    const dashOffset = circumference * (1 - progressRatio);

    const latest = snap?.rounds_history?.[snap.rounds_history.length - 1];

    let html = `
      <defs>
        <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10" />
        </filter>
      </defs>
    `;

    // Links
    nodes.forEach((n) => {
      const c = snap?.clients?.find((x) => x.cid === n.cid);
      const st = c?.status || "idle";
      const hasSubmitted = receivedSet.has(n.cid);
      const stroke =
        hasSubmitted ? "#059669" : st === "training" ? "#f59e0b" : st === "polling" ? "#3b82f6" : "#cbd5e1";
      const width = hasSubmitted || st === "training" || st === "sending" ? 2.4 : 1.6;
      const dash = st === "idle" || st === "waiting" ? 'stroke-dasharray="5 5"' : "";
      html += `<line x1="${cx}" y1="${cy}" x2="${n.x.toFixed(1)}" y2="${n.y.toFixed(1)}" stroke="${stroke}" stroke-width="${width}" ${dash} opacity="0.85" />`;
    });

    // Packet layer placeholder
    html += `<g id="packet-layer"></g>`;

    // Central Server Node
    const serverColor = snap?.status === "completed" ? "#059669" : "#2563eb";
    html += `
      <g transform="translate(${cx}, ${cy})" filter="url(#soft-shadow)">
        <circle r="58" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />
        <circle r="${ringRadius}" fill="none" stroke="#e2e8f0" stroke-width="5" />
        <circle r="${ringRadius}" fill="none" stroke="${serverColor}" stroke-width="5"
          stroke-dasharray="${circumference.toFixed(1)}"
          stroke-dashoffset="${dashOffset.toFixed(1)}"
          stroke-linecap="round"
          transform="rotate(-90)" />
        <text y="-14" text-anchor="middle" font-size="11" font-weight="700" fill="#64748b">SERVIDOR gRPC</text>
        <text y="5" text-anchor="middle" font-size="15" font-weight="700" font-family="Consolas, monospace" fill="#0f172a">
          R${snap ? Math.max(0, snap.rounds_history.length - 1) : 0}/${snap?.config?.total_rounds ?? 10}
        </text>
        <text y="22" text-anchor="middle" font-size="10.5" font-family="Consolas, monospace" fill="${serverColor}">
          ${snap?.status === "completed" ? `Acc ${fmtPct(latest?.global_acc)}` : `Buffer ${receivedCount}/${numClients}`}
        </text>
      </g>
    `;

    // Client Nodes
    nodes.forEach((n) => {
      const c = snap?.clients?.find((x) => x.cid === n.cid);
      const meta = CLIENT_STATUS[c?.status || "idle"] || CLIENT_STATUS.idle;
      const accText = c && c.last_acc != null ? `Acc ${(c.last_acc * 100).toFixed(0)}%` : c ? `${c.train_examples} ex.` : "Pronto";

      html += `
        <g transform="translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})" filter="url(#soft-shadow)">
          <rect x="-54" y="-29" width="108" height="58" rx="11" fill="#ffffff" stroke="${meta.color}" stroke-width="2" />
          <circle cx="-39" cy="-12" r="4.5" fill="${meta.color}" />
          <text x="-29" y="-8" font-size="11.5" font-weight="700" fill="#0f172a">Cliente ${n.cid}</text>
          <text x="0" y="9" text-anchor="middle" font-size="10.5" font-weight="600" fill="${meta.color}">${meta.label}</text>
          <text x="0" y="22" text-anchor="middle" font-size="10" font-family="Consolas, monospace" fill="#64748b">${accText}</text>
        </g>
      `;
    });

    svg.innerHTML = html;
    drawPackets();
  }

  function drawPackets() {
    const layer = $("#packet-layer");
    if (!layer) return;
    const now = performance.now();
    state.packets = state.packets.filter((p) => now - p.born < p.duration);
    layer.innerHTML = state.packets
      .map((p) => {
        const t = Math.min(1, (now - p.born) / p.duration);
        const ease = 1 - Math.pow(1 - t, 2);
        const x = p.x1 + (p.x2 - p.x1) * ease;
        const y = p.y1 + (p.y2 - p.y1) * ease;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5" fill="${p.color}" stroke="#ffffff" stroke-width="1.8" />`;
      })
      .join("");
  }

  function animationLoop() {
    if (state.packets.length > 0) {
      drawPackets();
    }
    requestAnimationFrame(animationLoop);
  }
  requestAnimationFrame(animationLoop);

  // ------------------------------------------------------------------
  // Dual SVG Charts (Accuracy & Log-Loss)
  // ------------------------------------------------------------------
  function renderCharts() {
    const snap = state.snapshot;
    const rounds = snap?.rounds_history || [];
    const baseline = snap?.manual_baseline?.rounds || [];

    renderLineChart("#chart-acc", {
      series: [
        { data: rounds.map((r) => ({ x: r.round, y: r.global_acc })), color: "#2563eb", dashed: false },
        { data: rounds.filter((r) => r.avg_train_acc != null).map((r) => ({ x: r.round, y: r.avg_train_acc })), color: "#059669", dashed: false },
        { data: baseline.map((r) => ({ x: r.round, y: r.global_acc })), color: "#d97706", dashed: true },
      ],
      yMin: 0,
      yMax: 1.0,
      yFormat: (v) => `${Math.round(v * 100)}%`,
      maxRound: Math.max(snap?.config?.total_rounds || 10, rounds.length - 1, baseline.length - 1),
    });

    const allLosses = [
      ...rounds.map((r) => r.global_loss),
      ...rounds.map((r) => r.avg_train_loss),
      ...baseline.map((r) => r.global_loss),
    ].filter((v) => v != null);
    const maxLoss = allLosses.length ? Math.max(1.2, Math.ceil(Math.max(...allLosses) * 10) / 10) : 1.2;

    renderLineChart("#chart-loss", {
      series: [
        { data: rounds.map((r) => ({ x: r.round, y: r.global_loss })), color: "#2563eb", dashed: false },
        { data: rounds.filter((r) => r.avg_train_loss != null).map((r) => ({ x: r.round, y: r.avg_train_loss })), color: "#059669", dashed: false },
        { data: baseline.map((r) => ({ x: r.round, y: r.global_loss })), color: "#d97706", dashed: true },
      ],
      yMin: 0,
      yMax: maxLoss,
      yFormat: (v) => v.toFixed(2),
      maxRound: Math.max(snap?.config?.total_rounds || 10, rounds.length - 1, baseline.length - 1),
    });
  }

  function renderLineChart(selector, { series, yMin, yMax, yFormat, maxRound }) {
    const svg = $(selector);
    const W = 520;
    const H = 190;
    const pad = { top: 14, right: 16, bottom: 28, left: 46 };
    const pw = W - pad.left - pad.right;
    const ph = H - pad.top - pad.bottom;

    const xScale = (x) => pad.left + (x / Math.max(1, maxRound)) * pw;
    const yScale = (y) => pad.top + ph - ((y - yMin) / Math.max(1e-6, yMax - yMin)) * ph;

    let out = "";

    // Horizontal gridlines
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + ((yMax - yMin) * i) / yTicks;
      const y = yScale(val);
      out += `<line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="#e2e8f0" stroke-width="1" />`;
      out += `<text x="${pad.left - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-size="10" font-family="Consolas, monospace" fill="#64748b">${yFormat(val)}</text>`;
    }

    // X ticks
    const step = maxRound > 10 ? 2 : 1;
    for (let r = 0; r <= maxRound; r += step) {
      const x = xScale(r);
      out += `<text x="${x.toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" font-family="Consolas, monospace" fill="#64748b">R${r}</text>`;
    }

    const hasAny = series.some((s) => s.data.length > 0);
    if (!hasAny) {
      out += `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" font-size="12" fill="#94a3b8">Aguardando dados da simulação...</text>`;
      svg.innerHTML = out;
      return;
    }

    series.forEach((s) => {
      if (!s.data.length) return;
      const pts = s.data.map((p) => `${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`).join(" ");
      const dash = s.dashed ? 'stroke-dasharray="5 4"' : "";
      out += `<polyline fill="none" stroke="${s.color}" stroke-width="2.3" ${dash} points="${pts}" />`;
      s.data.forEach((p) => {
        out += `<circle cx="${xScale(p.x).toFixed(1)}" cy="${yScale(p.y).toFixed(1)}" r="3.2" fill="#ffffff" stroke="${s.color}" stroke-width="2">
          <title>Rodada ${p.x}: ${yFormat(p.y)}</title>
        </circle>`;
      });
    });

    svg.innerHTML = out;
  }

  // ------------------------------------------------------------------
  // Rounds Table
  // ------------------------------------------------------------------
  function renderRoundsTable() {
    const tbody = $("#rounds-table-body");
    const rounds = state.snapshot?.rounds_history || [];
    const baseline = state.snapshot?.manual_baseline?.rounds || [];
    const baselineMap = new Map(baseline.map((b) => [b.round, b]));

    if (!rounds.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-cell">Inicie uma sessão gRPC para acompanhar a convergência rodada a rodada.</td></tr>`;
      return;
    }

    tbody.innerHTML = rounds
      .slice()
      .reverse()
      .map((r) => {
        const b = baselineMap.get(r.round);
        return `
          <tr>
            <td class="mono"><strong>Rodada ${r.round}</strong></td>
            <td class="mono">${fmtNum(r.avg_train_loss, 4)}</td>
            <td class="mono">${fmtPct(r.avg_train_acc)}</td>
            <td class="mono">${fmtNum(r.global_loss, 4)}</td>
            <td class="mono accent-emerald"><strong>${fmtPct(r.global_acc)}</strong></td>
            <td class="mono">${b ? `${fmtPct(b.global_acc)} (loss ${fmtNum(b.global_loss, 4)})` : "—"}</td>
          </tr>
        `;
      })
      .join("");
  }

  // ------------------------------------------------------------------
  // Clients & Partitions Grid
  // ------------------------------------------------------------------
  function renderClientsGrid() {
    const grid = $("#clients-grid");
    const clients = state.snapshot?.clients || [];
    if (!clients.length) {
      grid.innerHTML = `<div class="empty-state">Inicie uma sessão gRPC para visualizar as partições locais dos clientes.</div>`;
      return;
    }

    const defaultOrder = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

    grid.innerHTML = clients
      .map((c) => {
        const st = CLIENT_STATUS[c.status] || CLIENT_STATUS.idle;
        const dist = c.train_distribution || {};
        const total = Math.max(1, c.train_examples);
        const keys = defaultOrder.some((k) => dist[k] != null) ? defaultOrder : Object.keys(dist);
        const segs = keys
          .map((sp, i) => {
            const count = dist[sp] || 0;
            const w = ((count / total) * 100).toFixed(1);
            return `<div class="dist-seg-${i % 3}" style="width:${w}%" title="${sp}: ${count} amostras"></div>`;
          })
          .join("");

        return `
          <div class="client-tile">
            <div class="client-tile-head">
              <span class="client-title">Cliente ${c.cid}</span>
              <span class="pill ${st.pill}">${st.label}</span>
            </div>
            <div>
              <div class="dist-bar">${segs}</div>
              <div class="muted-text mono" style="font-size:0.72rem; margin-top:4px;">
                Treino: ${c.train_examples} | Teste: ${c.test_examples} | Peso FedAvg: ${(c.fedavg_weight * 100).toFixed(1)}%
              </div>
            </div>
            <div class="client-metrics">
              <div class="metric-mini">
                <span>Última Rodada</span>
                <strong>R${c.completed_round}</strong>
              </div>
              <div class="metric-mini">
                <span>Acurácia Local</span>
                <strong>${fmtPct(c.last_acc)}</strong>
              </div>
              <div class="metric-mini">
                <span>Loss Local</span>
                <strong>${fmtNum(c.last_loss, 4)}</strong>
              </div>
              <div class="metric-mini">
                <span>Classes (Treino)</span>
                <strong>${Object.values(dist).join(" / ") || "—"}</strong>
              </div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // ------------------------------------------------------------------
  // Weights Heatmap
  // ------------------------------------------------------------------
  function renderWeightsHeatmap() {
    const history = state.snapshot?.weights_history || [];
    const slider = $("#weights-round-slider");
    const wrap = $("#weights-heatmap");
    const meta = $("#weights-meta");

    if (!history.length) {
      slider.max = 0;
      slider.value = 0;
      $("#weights-round-label").textContent = "0";
      meta.innerHTML = "";
      wrap.innerHTML = `<div class="empty-state">Nenhum parâmetro carregado ainda.</div>`;
      return;
    }

    slider.max = history.length - 1;
    const idx =
      state.weightsRoundIndex === null || state.weightsRoundIndex >= history.length
        ? history.length - 1
        : state.weightsRoundIndex;
    slider.value = idx;
    const snap = history[idx];
    $("#weights-round-label").textContent = String(snap.round);

    meta.innerHTML = `
      <span>Norma L2 global: <strong>${snap.l2_norm.toFixed(4)}</strong></span>
      <span>Δ desde rodada anterior: <strong>${snap.delta_norm.toFixed(4)}</strong></span>
      <span>Shape coef_: <strong>(${snap.classes.length}, ${snap.features.length})</strong></span>
    `;

    const allVals = [...snap.coef.flat(), ...snap.intercept];
    const maxAbs = Math.max(0.01, ...allVals.map((v) => Math.abs(v)));

    function cellStyle(v) {
      const intensity = Math.min(1, Math.abs(v) / maxAbs);
      if (Math.abs(v) < 1e-6) return "background:#f8fafc; color:#64748b;";
      if (v > 0) {
        const a = (0.1 + intensity * 0.55).toFixed(2);
        return `background:rgba(37, 99, 235, ${a}); color:${intensity > 0.55 ? "#ffffff" : "#1e3a8a"};`;
      }
      const a = (0.1 + intensity * 0.55).toFixed(2);
      return `background:rgba(220, 38, 38, ${a}); color:${intensity > 0.55 ? "#ffffff" : "#7f1d1d"};`;
    }

    let table = `
      <table class="heatmap-table">
        <thead>
          <tr>
            <th class="row-label">Espécie</th>
            ${snap.features.map((f) => `<th>${f}</th>`).join("")}
            <th>intercept_</th>
          </tr>
        </thead>
        <tbody>
    `;

    snap.classes.forEach((cls, r) => {
      table += `<tr><th class="row-label">${escapeHtml(cls)}</th>`;
      snap.coef[r].forEach((val) => {
        table += `<td class="heatmap-cell" style="${cellStyle(val)}">${val >= 0 ? "+" : ""}${val.toFixed(3)}</td>`;
      });
      const b = snap.intercept[r];
      table += `<td class="heatmap-cell" style="${cellStyle(b)}">${b >= 0 ? "+" : ""}${b.toFixed(3)}</td></tr>`;
    });

    table += `</tbody></table>`;
    wrap.innerHTML = table;
  }

  // ------------------------------------------------------------------
  // Confusion Matrix
  // ------------------------------------------------------------------
  function renderConfusionMatrix() {
    const wrap = $("#confusion-wrap");
    const cm = state.snapshot?.confusion_matrix;
    if (!cm || !cm.matrix) {
      wrap.innerHTML = `<div class="empty-state">Execute uma sessão gRPC para gerar a matriz de confusão.</div>`;
      return;
    }

    const maxVal = Math.max(1, ...cm.matrix.flat());
    let html = `
      <table class="confusion-table">
        <thead>
          <tr>
            <th>Real \\ Previsto</th>
            ${cm.labels.map((l) => `<th>${escapeHtml(l.replace("Iris-", ""))}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
    `;

    cm.matrix.forEach((row, i) => {
      html += `<tr><th>${escapeHtml(cm.labels[i].replace("Iris-", ""))}</th>`;
      row.forEach((val, j) => {
        const ratio = val / maxVal;
        const isDiag = i === j;
        const bg =
          val === 0
            ? "#f8fafc"
            : isDiag
              ? `rgba(5, 150, 105, ${(0.18 + ratio * 0.65).toFixed(2)})`
              : `rgba(220, 38, 38, ${(0.16 + ratio * 0.55).toFixed(2)})`;
        const fg = val === 0 ? "#94a3b8" : ratio > 0.55 ? "#ffffff" : isDiag ? "#065f46" : "#991b1b";
        html += `<td class="conf-cell" style="background:${bg}; color:${fg};">${val}</td>`;
      });
      html += `</tr>`;
    });

    html += `</tbody></table>`;
    wrap.innerHTML = html;
  }

  // ------------------------------------------------------------------
  // RPC Log Table
  // ------------------------------------------------------------------
  function renderRpcTable() {
    const tbody = $("#rpc-table-body");
    const logs = (state.snapshot?.rpc_logs || []).slice().reverse();
    const filtered = logs.filter((entry) => {
      if (state.rpcFilter === "all") return true;
      if (state.rpcFilter === "rejected") return !entry.accepted;
      return entry.method === state.rpcFilter;
    });

    if (!filtered.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="empty-cell">Nenhuma chamada gRPC corresponde ao filtro atual.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered
      .map((e) => {
        const statusPill = !e.accepted
          ? `<span class="pill pill-red">RECUSADO</span>`
          : e.done
            ? `<span class="pill pill-emerald">DONE</span>`
            : `<span class="pill pill-blue">OK</span>`;
        const methodColor = e.method === "GetGlobalModel" ? "#2563eb" : "#059669";
        return `
          <tr>
            <td>#${e.id}</td>
            <td>${e.time}</td>
            <td style="color:${methodColor}; font-weight:700;">${e.method}</td>
            <td>cid=${e.cid}</td>
            <td>R${e.round}</td>
            <td>${e.req_bytes}B / ${e.resp_bytes}B</td>
            <td>${e.latency_ms.toFixed(2)} ms</td>
            <td>${statusPill}</td>
            <td>${escapeHtml(e.message)}</td>
          </tr>
        `;
      })
      .join("");
  }

  // ------------------------------------------------------------------
  // Unit Tests Renderer
  // ------------------------------------------------------------------
  function renderTestResults(res) {
    const summary = $("#tests-summary");
    const list = $("#tests-list");
    summary.classList.remove("hidden");
    summary.style.background = res.success ? "#ecfdf5" : "#fef2f2";
    summary.style.color = res.success ? "#065f46" : "#991b1b";
    summary.style.border = `1px solid ${res.success ? "#a7f3d0" : "#fecaca"}`;
    summary.textContent = res.success
      ? `✓ Sucesso: ${res.passed_cases}/${res.total_cases} testes passaram em ${res.total_ms} ms`
      : `✕ Falha: ${res.passed_cases}/${res.total_cases} testes passaram em ${res.total_ms} ms`;

    if (!res.modules || !res.modules.length) {
      list.innerHTML = `<pre class="inject-result">${escapeHtml(res.stderr || "Sem detalhes.")}</pre>`;
      return;
    }

    list.innerHTML = res.modules
      .map(
        (m) => `
        <div class="test-module-card">
          <div class="test-module-head">
            <span class="mono">${escapeHtml(m.module)}</span>
            <span class="pill ${m.passed ? "pill-emerald" : "pill-red"} mono">
              ${m.passed ? "PASSOU" : "FALHOU"} • ${m.ran} testes • ${m.duration_ms} ms
            </span>
          </div>
          <ul class="test-cases">
            ${m.cases
              .map(
                (c) => `
              <li>
                <span>${c.passed ? "✓" : "✕"} ${escapeHtml(c.name)}</span>
                <span class="pill ${c.passed ? "pill-emerald" : "pill-red"}">${c.passed ? "OK" : "FAIL"}</span>
              </li>`
              )
              .join("")}
          </ul>
        </div>
      `
      )
      .join("");
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  async function init() {
    renderTopology();
    renderCharts();
    connectSSE();
    try {
      const resp = await fetch("/api/state");
      if (resp.ok) {
        applySnapshot(await resp.json());
      }
    } catch (err) {
      console.error(err);
    }
  }

  init();
})();
