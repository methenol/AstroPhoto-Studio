/* Jobs tab: the server's queue (running / paused / waiting), other work on the machine, history */
const Jobs = (() => {
  const J = { timer: 0, open: new Set() };
  const STATE = { running: "running", paused: "paused", queued: "waiting", done: "done", error: "failed", cancelled: "cancelled", interrupted: "interrupted" };
  const when = t => t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–";

  function card(j, history) {
    const ctl = [];
    if (j.state === "running") ctl.push(`<button class="btn tiny" data-act="pause" data-id="${j.id}">Pause</button>`);
    if (j.state === "paused") ctl.push(`<button class="btn tiny" data-act="resume" data-id="${j.id}">Resume</button>`);
    if (["running", "paused"].includes(j.state)) ctl.push(`<button class="btn tiny ghost" data-act="cancel" data-id="${j.id}">Cancel</button>`);
    if (j.state === "queued") ctl.push(`<button class="btn tiny ghost" data-act="cancel" data-id="${j.id}">Remove from queue</button>`);
    if (history) ctl.push(`<button class="btn tiny ghost" data-act="rerun" data-id="${j.id}">Run again</button>`);
    const stages = (j.stages || []).length ? `<details class="small" ${J.open.has(j.id) ? "open" : ""} data-det="${j.id}"><summary class="muted">Stages</summary>
      <table class="jobstages"><tbody>${j.stages.map(([k, t]) => `<tr><td>${esc(k)}</td><td class="num">${fmtDur(t)}</td></tr>`).join("")}</tbody></table></details>` : "";
    const err = j.state === "error" ? `<details class="small"><summary style="color:var(--bad)">${esc(j.message)}</summary>
      <pre class="log" style="max-height:300px;user-select:text">${esc((j.device_state ? j.device_state + "\n\n" : "") + (j.traceback || ""))}</pre></details>` : "";
    const bar = history ? "" : `<div class="progress"><div style="width:${(j.progress * 100).toFixed(1)}%;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent))"></div></div>`;
    const times = history
      ? `${when(j.created)} · ${j.state === "done" ? "took " + fmtDur(j.active_seconds ?? j.active) : esc(j.message)}${j.paused_total ? ` (+${fmtDur(j.paused_total)} paused)` : ""}`
      : jobSummary(j);
    return `<div class="jobcard ${esc(j.state)}">
      <div class="jobhead"><b>${esc(j.label)}</b><span class="muted">${esc(j.dataset || "")}${j.n_subs ? ` · ${j.n_subs} subs` : ""}</span>
        <span class="state ${esc(j.state)}">${STATE[j.state] || esc(j.state)}</span><span class="jobctl">${ctl.join(" ")}</span></div>
      ${bar}<div class="small jobmsg">${times}</div>${stages}${err}</div>`;
  }

  async function refresh() {
    let r;
    try { r = await api("/api/jobs"); } catch (e) { return; }
    badge(r);
    if (!$("#tab-jobs").classList.contains("active")) return;
    const busy = r.active.find(j => j.state === "running" || j.state === "paused");
    const waiting = r.active.filter(j => j.state === "queued").length;
    $("#jobsServer").textContent = busy ? `busy: ${busy.label} on ${busy.dataset}${waiting ? `, ${waiting} waiting` : ""}` : (waiting ? `${waiting} waiting` : "idle");
    $("#jobsActive").innerHTML = r.active.length ? r.active.map(j => card(j, false)).join("")
      : `<p class="muted small">Nothing is running. Jobs started from the Pipeline card appear here, and wait their turn when the server is busy.</p>`;
    $("#jobsExternalCard").hidden = !r.external.length;
    $("#jobsExternal").innerHTML = r.external.map(x => `<div class="jobcard"><div class="jobhead"><b>${esc(x.kind)}</b><span class="muted">${esc(x.name || x.id)}${x.device ? " · device " + esc(x.device) : ""}</span>
      <span class="state running">${esc(x.state)}</span></div><div class="small jobmsg">${esc(x.message || "")}${x.trial !== undefined && x.trial !== null ? ` · trial ${x.trial + 1}${x.n_trials ? "/" + x.n_trials : ""}` : ""}${x.started ? ` · since ${when(x.started)}` : ""}
      — runs in its own process alongside the queue (manage it in the Experiments tab)</div></div>`).join("");
    $("#jobsHistory").innerHTML = r.history.length ? r.history.map(j => card(j, true)).join("") : `<p class="muted small">No finished jobs yet.</p>`;
    $$("#tab-jobs [data-act]").forEach(b => b.onclick = async () => {
      try {
        const res = await api(`/api/jobs/${b.dataset.id}/${b.dataset.act}`, { method: "POST" });
        if (b.dataset.act === "rerun" && res.folder === S.folder) { S.job = res; $("#jobCard").hidden = false; pollJob(); }
        refresh();
      } catch (e) { toast(e.message, true); }
    });
    $$("#tab-jobs [data-det]").forEach(d => d.ontoggle = () => d.open ? J.open.add(d.dataset.det) : J.open.delete(d.dataset.det));
  }

  function badge(r) {
    const n = r.active.length;
    const b = $("#jobsCount");
    b.hidden = !n; b.textContent = n;
  }

  function show() { refresh(); }
  $("#jobsClear").onclick = async () => { await api("/api/jobs", { method: "DELETE" }); refresh(); };
  // the badge is kept up to date on every tab; the list refreshes while the tab is open
  (function loop() { refresh().finally(() => { J.timer = setTimeout(loop, $("#tab-jobs").classList.contains("active") ? 1500 : 5000); }); })();
  return { show };
})();
