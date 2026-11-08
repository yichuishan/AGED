"use strict";
const $ = (id) => document.getElementById(id);
const state = { boot: null, selected: null, cursor: 0, logs: [], job: null, loadedCheckpoint: null, browseTarget: null, browseParent: null, busy: false };
const statusNames = { idle: "等待启动", running: "训练运行中", stopping: "正在停止", completed: "训练完成", stopped: "已保存并停止", failed: "训练失败", terminated: "已终止" };
const colours = ["#188d7b", "#a3ba5b", "#568cc5", "#d7a153", "#9d7ca6", "#c37065"];
let pollBusy = false;

async function api(path, body) {
  const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-AGED-Token": state.boot?.token || "" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function message(text, error = false, scroll = true) { const el = $("form-message"); el.textContent = text; el.classList.remove("hidden"); el.classList.toggle("error", error); if (scroll) requestAnimationFrame(() => el.scrollIntoView({ block: "nearest", behavior: "smooth" })); }
function number(value, digits = 4) { return Number.isFinite(value) ? (Math.abs(value) >= 1e5 ? value.toExponential(2) : value.toLocaleString("en-US", { maximumFractionDigits: digits })) : "—"; }
function element(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }

function buildForm() {
  const container = $("parameter-groups"), groups = new Map();
  state.boot.schema.forEach((field) => {
    if (!groups.has(field.group)) {
      const details = element("details", undefined, "parameter-group"); details.open = groups.size === 0;
      details.append(element("summary", field.group));
      const grid = element("div", undefined, "group-fields"); details.append(grid); container.append(details); groups.set(field.group, grid);
    }
    const wrap = element("div", undefined, "field");
    const label = element("label", field.label); label.htmlFor = "param-" + field.name; label.append(element("code", field.name));
    if (field.kind === "boolean") {
      const row = element("div", undefined, "boolean-field"), input = element("input"); input.type = "checkbox"; input.id = "param-" + field.name; row.append(input, label); wrap.append(row);
    } else if (field.kind === "actions") {
      wrap.classList.add("wide"); wrap.append(label);
      const allLabel = element("label", undefined, "action-select-all"), all = element("input"); all.type = "checkbox"; all.id = "actions-all"; all.checked = true;
      all.addEventListener("change", () => document.querySelectorAll("[data-action]").forEach((input) => { input.checked = all.checked; }));
      allLabel.append(all, document.createTextNode("全部动作")); wrap.append(allLabel);
      const list = element("div", undefined, "actions-box");
      state.boot.actions.forEach((action) => { const row = element("label"), input = element("input"); input.type = "checkbox"; input.dataset.action = action; input.checked = true; input.setAttribute("aria-label", action); row.append(input, document.createTextNode(action)); list.append(row); });
      wrap.append(list);
    } else {
      wrap.append(label);
      let input;
      if (field.kind === "select") {
        input = element("select"); field.choices.forEach((choice) => { const option = element("option", choice === null ? (field.name === "val_subject" ? "全部六人 · 不设验证集" : "跟随 G") : String(choice)); option.value = choice === null ? "" : String(choice); input.append(option); });
      } else {
        input = element("input"); input.type = ["number", "integer"].includes(field.kind) ? "number" : "text";
        if (input.type === "number") { input.step = field.kind === "integer" ? "1" : "any"; input.min = "0"; }
        if (field.nullable) input.placeholder = "跟随 G"; else input.required = true;
      }
      input.id = "param-" + field.name; wrap.append(input);
    }
    if (field.help) wrap.append(element("small", field.help));
    groups.get(field.group).append(wrap);
  });
}
function setConfig(config) {
  state.boot.schema.forEach((field) => {
    if (field.kind === "actions") document.querySelectorAll("[data-action]").forEach((input) => { input.checked = config.actions.includes(input.dataset.action); });
    else if (field.kind === "boolean") $("param-" + field.name).checked = config[field.name];
    else $("param-" + field.name).value = config[field.name] ?? "";
  });
  updateLocks();
}
function getConfig() {
  const config = {};
  state.boot.schema.forEach((field) => {
    const input = $("param-" + field.name);
    if (field.kind === "actions") config.actions = [...document.querySelectorAll("[data-action]:checked")].map((el) => el.dataset.action);
    else if (field.kind === "boolean") config[field.name] = input.checked;
    else if (field.kind === "select") config[field.name] = input.value || null;
    else if (field.kind === "number" || field.kind === "integer") {
      if (!input.value && !field.nullable) throw new Error(`${field.label}不能为空`);
      config[field.name] = !input.value && field.nullable ? null : Number(input.value);
      if (config[field.name] !== null && !Number.isFinite(config[field.name])) throw new Error(`${field.label}必须是有限数值`);
    } else config[field.name] = input.value;
  });
  return config;
}
function payload() { return { mode: $("mode").value, root: $("root").value, out: $("out").value, checkpoint: $("checkpoint").value, config: getConfig() }; }
function updateLocks() {
  const resume = $("mode").value === "resume";
  state.boot.schema.forEach((field) => {
    const locked = resume && !field.resume_mutable;
    if (field.kind === "actions") document.querySelectorAll("[data-action],#actions-all").forEach((input) => { input.disabled = locked || state.busy; });
    else $("param-" + field.name).disabled = locked || state.busy;
  });
  ["mode", "preset", "root", "out", "checkpoint", "load-checkpoint", "validate", "start"].forEach((id) => { $(id).disabled = state.busy || (id === "preset" && resume); });
  document.querySelectorAll("[data-browse]").forEach((button) => { button.disabled = state.busy; });
  $("checkpoint-fields").classList.toggle("hidden", $("mode").value === "new");
}
function newOutput() { return "experiments/web-" + new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19); }
function usePreset() {
  setConfig(state.boot.profiles[$("preset").value].config);
  const notes = { p0: "建议先验证 SGD + 0.005：S11 开发集，2 × 1000 次更新，每轮保留检查点。", short: "保留原默认 Adam + 0.005；组内旧 GPU 记录显示该组合不稳定，请先做短跑。", long: "独立 25 帧长期模型；保留 Adam 默认，正式训练前请先验证稳定性。", team: "团队组合同时改变多项设置，并使用 non-saturating 目标；先完成单项对照，再测此组合。" };
  $("preset-note").textContent = notes[$("preset").value];
}

function chart(id, rows, xKey, definitions, empty) {
  const host = $(id); host.replaceChildren();
  const series = definitions.map((def, index) => ({ ...def, colour: colours[index], points: rows.map((row) => ({ x: row[xKey], y: def.value(row) })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)) })).filter((s) => s.points.length);
  if (!series.length) { host.append(element("div", empty, "chart-empty")); return; }
  const ns = "http://www.w3.org/2000/svg";
  function svgEl(name, attrs = {}, text) { const el = document.createElementNS(ns, name); Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value)); if (text !== undefined) el.textContent = text; return el; }
  const svg = svgEl("svg", { viewBox: "0 0 400 180", role: "img", "aria-label": id === "loss-chart" ? "训练损失曲线" : "验证误差曲线" });
  const all = series.flatMap((s) => s.points), xmin = Math.min(...all.map((p) => p.x)), xmax = Math.max(xmin + 1, ...all.map((p) => p.x));
  let ymin = Math.min(...all.map((p) => p.y)), ymax = Math.max(...all.map((p) => p.y));
  const pad = Math.max((ymax - ymin) * 0.12, Math.abs(ymax) * 0.03, 0.001); ymin -= pad; ymax += pad;
  const x = (v) => 43 + (v - xmin) / (xmax - xmin) * 344, y = (v) => 148 - (v - ymin) / (ymax - ymin) * 130;
  for (let i = 0; i < 4; i++) {
    const value = ymin + (ymax - ymin) * i / 3, at = y(value);
    svg.append(svgEl("line", { x1: 43, x2: 387, y1: at, y2: at, stroke: "#edf1f3", "stroke-width": 1 }), svgEl("text", { x: 36, y: at + 3, "text-anchor": "end", fill: "#9babad", "font-size": 9 }, number(value, 2)));
  }
  [xmin, xmax].forEach((value) => svg.append(svgEl("text", { x: x(value), y: 168, "text-anchor": value === xmin ? "start" : "end", fill: "#9babad", "font-size": 9 }, number(value, 0))));
  series.forEach((s) => {
    svg.append(svgEl("path", { d: s.points.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(" "), fill: "none", stroke: s.colour, "stroke-width": 2, "stroke-linejoin": "round" }));
    s.points.forEach((p) => { const dot = svgEl("circle", { cx: x(p.x), cy: y(p.y), r: s.points.length < 20 ? 3 : 1.5, fill: s.colour }); dot.append(svgEl("title", {}, `${s.label} · ${xKey} ${p.x}: ${number(p.y)}`)); svg.append(dot); });
  });
  host.append(svg);
  const legend = element("div", undefined, "chart-legend"); series.forEach((s) => { const label = element("span"), mark = element("i"); mark.style.backgroundColor = s.colour; label.append(mark, document.createTextNode(s.label)); legend.append(label); }); host.append(legend);
}
function renderLogs() {
  const consoleEl = $("console"); consoleEl.replaceChildren();
  const isError = (line) => /^\s*[\w.]*(?:Error|Exception):|Traceback \(most recent call last\)|错误|失败|nonfinite|out of memory/i.test(line);
  const rows = $("errors-only").checked ? state.logs.filter((row) => isError(row.text)) : state.logs;
  if (!rows.length) consoleEl.append(element("span", $("errors-only").checked ? "暂无匹配的错误行；完整堆栈可在全部日志查看。" : "任务已启动，正在等待训练进程输出…", "console-placeholder"));
  rows.forEach((row) => consoleEl.append(element("span", row.text + "\n", isError(row.text) ? "log-error" : "")));
  $("log-count").textContent = `${state.logs.length} 行`;
  if ($("auto-scroll").checked) consoleEl.scrollTop = consoleEl.scrollHeight;
}
function renderJob(job) {
  state.job = job;
  const active = ["running", "stopping"].includes(job.status), progress = job.progress, last = job.history.at(-1);
  const metrics = active ? progress?.metrics : last || progress?.metrics;
  const epoch = progress?.epoch || last?.epoch || 0;
  $("status").className = "status " + job.status; $("status").lastElementChild.textContent = statusNames[job.status] || job.status;
  $("epoch-metric").replaceChildren(document.createTextNode(String(epoch)), element("small", ` / ${job.config.epochs} 轮`));
  const step = progress?.step || last?.step || 0;
  $("step-metric").textContent = `${number(step, 0)} / ${number(job.config.epochs * job.config.steps_per_epoch, 0)} 次更新`;
  $("progress-bar").style.width = Math.min(100, step / (job.config.epochs * job.config.steps_per_epoch) * 100) + "%";
  $("loss-metric").textContent = number(metrics?.reconstruction); $("loss-caption").textContent = active ? "最近训练 batch · 即时值" : "最近完成 epoch · 平均值";
  const errors = last?.validation_angle_error;
  $("validation-metric").textContent = errors ? number(Object.values(errors).reduce((a, b) => a + b, 0) / Object.keys(errors).length) : "—";
  $("fidelity-metric").textContent = Number.isFinite(metrics?.fidelity_accuracy) ? number(metrics.fidelity_accuracy * 100, 1) + "%" : "—";
  $("continuity-metric").textContent = Number.isFinite(metrics?.continuity_accuracy) ? number(metrics.continuity_accuracy * 100, 1) + "%" : "—";
  $("gradient-metric").textContent = number(metrics?.generator_grad_norm, 2);
  chart("loss-chart", job.series, "step", [{ label: "Reconstruction", value: (r) => r.reconstruction }, { label: "Generator total", value: (r) => r.generator_loss }], "数据加载 / 准备模型时尚无曲线");
  const horizons = [...new Set(job.history.flatMap((r) => Object.keys(r.validation_angle_error || {})))];
  chart("validation-chart", job.history, "epoch", horizons.map((ms) => ({ label: ms + " ms", value: (r) => r.validation_angle_error?.[ms] })), job.config.val_subject ? "首轮验证完成后显示曲线" : "本实验未设置开发验证集");
  const newLogs = job.logs.filter((row) => row.id > state.cursor);
  if (newLogs.length) { state.logs.push(...newLogs); state.logs = state.logs.slice(-3000); state.cursor = job.cursor; renderLogs(); } else if (!state.logs.length) renderLogs();
  $("failure").classList.toggle("hidden", !job.error); $("failure").textContent = job.error ? "训练进程失败。以下为末尾异常；完整堆栈见日志。\n" + job.error : "";
  $("run-details").classList.remove("hidden"); $("run-out").textContent = job.out; $("command").textContent = job.command;
  $("checkpoints").replaceChildren(...job.checkpoints.map((name) => element("span", name, "checkpoint-tag")));
  $("download-log").classList.remove("hidden"); $("download-log").href = `/api/jobs/${job.id}/log`;
  $("stop").disabled = !active || job.status === "stopping"; $("force-stop").disabled = !active;
  $("stop-note").textContent = job.status === "stopping" ? "已请求停止；正常停止需等本轮验证和检查点保存完成。" : active ? "正常停止保留本轮；立即终止仅保留之前已完成的检查点。" : "任务已结束。可载入 last.pt 配置继续训练。";
}
async function refreshJobs() {
  const jobs = await api("/api/jobs"); state.busy = jobs.some((job) => ["running", "stopping"].includes(job.status)); updateLocks(); $("job-count").textContent = jobs.length;
  if (!jobs.length) return;
  $("job-list").replaceChildren();
  [...jobs].reverse().forEach((job) => { const button = element("button", job.out.split(/[\\/]/).pop(), "job-button" + (job.id === state.selected ? " active" : "")); button.type = "button"; button.append(element("small", statusNames[job.status] + " · " + new Date(job.created).toLocaleTimeString())); button.addEventListener("click", () => selectJob(job.id)); $("job-list").append(button); });
  if (!state.selected) await selectJob(jobs.at(-1).id);
}
async function selectJob(id) {
  state.selected = id; state.cursor = 0; state.logs = [];
  const job = await api(`/api/jobs/${id}`); if (id !== state.selected) return;
  $("mode").value = job.mode; $("checkpoint").value = job.checkpoint; state.loadedCheckpoint = null;
  setConfig(job.config); $("root").value = job.root; $("out").value = job.out;
  $("preset").value = "current";
  $("preset-note").textContent = "当前选中实验的实际配置。新实验可选择预设或修改参数，并使用新的输出目录。";
  renderJob(job); await refreshJobs();
}
async function poll() {
  if (!state.boot || pollBusy) return; pollBusy = true;
  try {
    if (state.selected) { const id = state.selected, job = await api(`/api/jobs/${id}?after=${state.cursor}`); if (id === state.selected) renderJob(job); }
    await refreshJobs(); $("connection-warning").classList.add("hidden");
  } catch (error) { $("connection-warning").textContent = "与本地服务连接中断：" + error.message + "。请检查运行网页服务器的终端。"; $("connection-warning").classList.remove("hidden"); }
  finally { pollBusy = false; }
}
async function browse(path) {
  try {
    const data = await api("/api/browse?path=" + encodeURIComponent(path || state.boot.environment.cwd));
    $("browse-path").value = data.path; state.browseParent = data.parent; $("browse-error").classList.add("hidden"); $("browse-entries").replaceChildren();
    const entries = data.entries.filter((entry) => state.browseTarget === "checkpoint" || entry.directory);
    if (!entries.length) $("browse-entries").append(element("p", "此目录下没有可选项；可手动输入完整路径。", "empty-directory"));
    entries.forEach((entry) => { const button = element("button", (entry.directory ? "▸  " : "◇  ") + entry.name, "file-entry"); button.type = "button"; button.addEventListener("click", () => { if (entry.directory) browse(entry.path); else { $(state.browseTarget).value = entry.path; $("browser-dialog").close(); state.loadedCheckpoint = null; } }); $("browse-entries").append(button); });
    $("browse-drives").replaceChildren(); data.drives.forEach((drive) => { const button = element("button", drive, "secondary"); button.type = "button"; button.addEventListener("click", () => browse(drive)); $("browse-drives").append(button); });
    $("browse-select").disabled = state.browseTarget === "checkpoint";
  } catch (error) { $("browse-error").textContent = error.message; $("browse-error").classList.remove("hidden"); }
}
function download(name, content) { const url = URL.createObjectURL(new Blob([content], { type: "application/json" })); const link = element("a"); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }

async function boot() {
  try {
    state.boot = await api("/api/bootstrap"); const env = state.boot.environment;
    $("device-name").textContent = env.gpu || "CPU 环境 · CUDA 不可用"; $("runtime-details").textContent = `Python ${env.python} / Torch ${env.torch}`; $("working-directory").textContent = env.cwd;
    Object.entries(state.boot.profiles).forEach(([key, preset]) => { const option = element("option", preset.label); option.value = key; $("preset").append(option); });
    const currentOption = element("option", "当前实验配置"); currentOption.value = "current"; currentOption.disabled = true; $("preset").append(currentOption);
    buildForm(); usePreset(); $("param-device").value = env.cuda ? "cuda" : "cpu";
    if (!env.cuda) message("当前环境没有 CUDA，已选择 CPU。完整训练请在 GPU 电脑启动同一网页服务。", false, false);
    $("out").value = newOutput();
    $("preset").addEventListener("change", () => { usePreset(); $("param-device").value = env.cuda ? "cuda" : "cpu"; });
    $("mode").addEventListener("change", () => { state.loadedCheckpoint = null; updateLocks(); }); $("checkpoint").addEventListener("input", () => { state.loadedCheckpoint = null; });
    $("load-checkpoint").addEventListener("click", async () => { try { const data = await api("/api/checkpoint", { path: $("checkpoint").value }); setConfig(data.config); $("param-epochs").value = Math.max(data.epoch + 1, data.config.epochs); $("param-device").value = env.cuda ? "cuda" : "cpu"; state.loadedCheckpoint = $("checkpoint").value; $("checkpoint-note").textContent = `已载入第 ${data.epoch} 轮 / ${data.step} 步。请确认总轮数、数据路径和输出目录。`; message("检查点配置已载入。续训时训练设置保持锁定。"); } catch (error) { message(error.message, true); } });
    $("validate").addEventListener("click", async () => { try { const result = await api("/api/validate", payload()); message(`检查通过：找到 ${result.files} 个训练/验证文件。文件内容与归一化将在启动时检查。\n输出：${result.out}`); } catch (error) { message(error.message, true); } });
    $("training-form").addEventListener("submit", async (event) => { event.preventDefault(); $("start").disabled = true; try { if ($("mode").value === "resume" && state.loadedCheckpoint !== $("checkpoint").value) throw new Error("请先载入检查点配置，再开始续训"); const job = await api("/api/start", payload()); message("训练已启动。首次读取数据时，请查看右侧实时日志。"); await selectJob(job.id); } catch (error) { message(error.message, true); } finally { updateLocks(); } });
    $("stop").addEventListener("click", async () => { try { renderJob(await api(`/api/jobs/${state.selected}/stop`, { force: false })); } catch (error) { message(error.message, true); } });
    $("force-stop").addEventListener("click", async () => { if (!confirm("立即终止会丢失本轮尚未保存的更新；之前的检查点保留。继续？")) return; try { renderJob(await api(`/api/jobs/${state.selected}/stop`, { force: true })); } catch (error) { message(error.message, true); } });
    document.querySelectorAll("[data-browse]").forEach((button) => button.addEventListener("click", () => { state.browseTarget = button.dataset.browse; $("browser-title").textContent = state.browseTarget === "checkpoint" ? "选择检查点文件" : "选择本机目录"; $("browser-dialog").showModal(); browse($(state.browseTarget).value); }));
    $("browse-parent").addEventListener("click", () => browse(state.browseParent)); $("browse-go").addEventListener("click", () => browse($("browse-path").value));
    $("browse-path").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); browse($("browse-path").value); } });
    $("browse-select").addEventListener("click", () => { $(state.browseTarget).value = $("browse-path").value; $("browser-dialog").close(); });
    document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
    $("config-preview").addEventListener("click", () => { try { $("config-json").textContent = JSON.stringify(getConfig(), null, 2); $("config-dialog").showModal(); } catch (error) { message(error.message, true); } });
    $("download-config").addEventListener("click", () => download("config.json", $("config-json").textContent + "\n")); $("errors-only").addEventListener("change", renderLogs); $("auto-scroll").addEventListener("change", renderLogs);
    await refreshJobs(); setInterval(poll, 1000);
  } catch (error) { $("connection-warning").textContent = "界面加载失败：" + error.message; $("connection-warning").classList.remove("hidden"); }
}
boot();
