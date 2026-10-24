/* Tools panel: Siril, GraXpert, StarNet, the RC-Astro licence (BlurXTerminator, NoiseXTerminator,
   StarXTerminator) and Cosmic Clarity (SetiAstro's models, run by siril-scripts' CosmicClarity_Native.py).  The server runs the tools; this only shows what /api/tools reports and sends the
   user's actions (activation code, activate / deactivate, model downloads). */
const Tools = (() => {
  const T = { data: null, open: new Set(["rcastro"]), busy: "", msg: {}, starnetOk: false, code: "" };

  const ok = t => !!(t && t.ok);
  const dot = cls => `<span class="dot-${cls}"></span>`;
  // the licence status classes: usable now / needs an action / not usable
  const pcls = p => (p.valid || ["trial", "licensed", "valid", "ok"].includes(p.status)) ? "ok"
    : ["available", "limit", "deactivated", "required"].includes(p.status) ? "warn" : "bad";

  async function load() {
    try { T.data = await api("/api/tools"); } catch (e) { T.data = null; $("#toolsBody").textContent = "Tools: " + e.message; return; }
    S.tools = T.data;
    render(); renderPlan();
  }

  /* which tool each processing step will use, from the licences and installed tools */
  function plan() {
    const d = T.data || {}, rc = d.rcastro || {}, prod = {};
    for (const p of rc.products || []) prod[p.key] = p;
    const live = k => prod[k] && pcls(prod[k]) === "ok" && ok(rc);
    const gx = d.graxpert || {}, sn = d.starnet || {}, cc = d.cosmic_clarity || {};
    const bge = ok(gx) && (!gx.models || gx.models.bge !== false);
    const ccOk = m => !!(cc.torch && cc.models && cc.models[m]);
    const sel = id => document.getElementById("sp-" + id)?.value;
    const deconvAuto = live("bxt") ? "BlurXTerminator (licensed)" : "Richardson-Lucy (Siril)";
    const denoiseAuto = live("nxt") ? "NoiseXTerminator (licensed)" : ok(gx) && gx.models?.denoise !== false ? "GraXpert denoising" : "Siril denoising";
    const starsAuto = live("sxt") ? "StarXTerminator (licensed)" : ok(sn) ? "StarNet2" : "none installed: the stars stay in the image";
    const dm = sel("deconv_method"), nm = sel("denoise_method"), sm = S.params?.star_removal;
    const gxOk = k => ok(gx) && (!gx.models || gx.models[k] !== false);
    const deconvNames = { bxt: "BlurXTerminator" + (live("bxt") ? "" : " (not licensed: nothing runs)"), rl: "Richardson-Lucy (Siril)", imagemm: "ImageMM", none: "off",
      graxpert_obj: "GraXpert deconvolution (objects)", graxpert_stellar: "GraXpert deconvolution (stars)", graxpert_both: "GraXpert deconvolution (objects + stars)" };
    const denoiseNames = { nxt: "NoiseXTerminator" + (live("nxt") ? "" : " (not licensed: nothing runs)"), graxpert: "GraXpert denoising", siril: "Siril denoising", none: "off" };
    const ccSharpen = S.params?.cc_sharpen && S.params.cc_sharpen !== "off", ccDenoise = !!S.params?.cc_denoise;
    return {
      background: bge ? "GraXpert background extraction" : "Siril background extraction (RBF)",
      deconv: (deconvNames[dm] || deconvAuto) + (ccSharpen ? (ccOk("sharpen") ? "; Cosmic Clarity Sharpen on the stretched image" : "; Cosmic Clarity Sharpen (models not installed)") : ""),
      denoise: (denoiseNames[nm] || denoiseAuto) + (ccDenoise ? (ccOk("denoise") ? "; Cosmic Clarity Denoise on the stretched image" : "; Cosmic Clarity Denoise (models not installed)") : ""),
      stars: sm === "darkstar" ? (ccOk("darkstar") ? "Cosmic Clarity DarkStar" : "Cosmic Clarity DarkStar (models not installed), then " + starsAuto) : starsAuto,
    };
  }
  function renderPlan() {
    const el = $("#toolPlan"); if (!el) return;
    if (!T.data) { el.textContent = "Checking the tools…"; return; }
    const p = plan();
    el.innerHTML = `Restoration will use: <b>${esc(p.background)}</b>, colour calibration (Siril SPCC), ` +
      `deconvolution <b>${esc(p.deconv)}</b>, noise reduction <b>${esc(p.denoise)}</b>; star removal <b>${esc(p.stars)}</b>.`;
  }

  const kv = rows => `<div class="tkv">${rows.filter(r => r[1] !== undefined && r[1] !== null && r[1] !== "").map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("")}</div>`;
  const note = key => T.msg[key] ? `<div class="tmsg ${T.msg[key].err ? "err" : ""}">${esc(T.msg[key].text)}</div>` : "";
  const card = (key, name, t, body, version) => `<details class="tool" data-tool="${key}" ${T.open.has(key) ? "open" : ""}>
    <summary>${dot(ok(t) ? "ok" : "bad")}<span class="tname">${name}</span><span class="tver">${ok(t) ? esc(version || "") : "not available"}</span></summary>
    <div class="tbody">${body}</div></details>`;

  function sirilCard(t) {
    t = t || {};
    return card("siril", "Siril", t, kv([["Version", esc(t.version)], ["Path", esc(t.path)]]) +
      (ok(t) ? "" : `<div class="tmsg err">${esc(t.error || t.reason || "siril-cli was not found: the stacking cannot run.")}</div>`), t.version);
  }

  function modelList(models) {
    const names = { bge: "Background extraction", denoise: "Denoising", deconv_obj: "Deconvolution (objects)", deconv_stellar: "Deconvolution (stars)" };
    return Object.entries(models || {}).map(([k, v]) => `<span class="pstat ${v ? "ok" : "warn"}" title="${esc(names[k] || k)}">${v ? "✓" : "–"} ${esc(names[k] || k)}</span>`).join(" ");
  }

  function graxpertCard(t) {
    t = t || {};
    const all = t.models && Object.values(t.models).every(Boolean);
    return card("graxpert", "GraXpert", t, kv([["Version", esc(t.version)], ["GPU", t.gpu === undefined ? "" : (t.gpu ? (typeof t.gpu === "string" ? esc(t.gpu) : "yes") : "no (CPU)")]]) +
      (t.models ? `<div>AI models: ${modelList(t.models)}</div>` : "") +
      (ok(t) ? `<div class="row gap"><button class="btn small" data-act="dl" data-tool="graxpert" ${T.busy ? "disabled" : ""}>${all ? "Check / update models" : "Download models"}</button></div>`
        : `<div class="tmsg err">${esc(t.error || t.reason || "GraXpert is not installed in this image.")}</div>`) +
      note("graxpert"), t.version);
  }

  function starnetCard(t) {
    t = t || {};
    const lic = "I accept the StarNet licence. The download comes from starnetastro.com and is covered by their licence agreement; it is stored in this server's data folder.";
    const need = !ok(t);
    return card("starnet", "StarNet2", t, kv([["Version", esc(t.version)], ["Path", esc(t.path)], ["Licence accepted", t.licence_accepted === undefined ? "" : (t.licence_accepted ? "yes" : "no")]]) +
      (need ? `<label class="check small" style="display:flex;gap:8px;align-items:flex-start"><input type="checkbox" id="starnetAccept" ${T.starnetOk ? "checked" : ""}> <span>${esc(lic)}</span></label>
        <div class="row gap"><button class="btn small" data-act="dl" data-tool="starnet" ${T.busy ? "disabled" : ""}>Download StarNet2</button></div>` : "") +
      note("starnet"), t.version);
  }

  function ccCard(t) {
    t = t || {};
    const names = { sharpen: "Sharpen", denoise: "Denoise", superres: "Super-resolution", darkstar: "DarkStar", satellite: "Satellite trails" };
    const models = Object.entries(t.models || {}).map(([k, v]) => `<span class="pstat ${v ? "ok" : "warn"}" title="${v ? "installed (" + v + ")" : "not installed"}">${v ? "✓" : "–"} ${esc(names[k] || k)}</span>`).join(" ");
    const any = Object.values(t.models || {}).some(Boolean);
    let body = kv([["Script", t.version ? "CosmicClarity_Native.py " + esc(t.version) : ""], ["PyTorch", t.torch === undefined ? "" : (t.torch ? "yes" : "no")],
                   ["onnxruntime", t.onnxruntime === undefined ? "" : (t.onnxruntime ? "yes" : "no")], ["Models", esc(t.models_dir)]]) +
      (t.models ? `<div>AI models: ${models}</div>` : "");
    if (t.torch === false) body += `<div class="tmsg err">${esc(t.reason || "This image has no PyTorch: build it with IMAGEMM=cuda (or cpu) to use Cosmic Clarity.")}</div>`;
    else if (t.script) body += `<div class="row gap"><button class="btn small" data-act="dl" data-tool="cosmic_clarity" ${T.busy ? "disabled" : ""}>${any ? "Download / update models" : "Download models (2.4 GB)"}</button></div>`;
    else body += `<div class="tmsg err">${esc(t.reason || "siril-scripts' CosmicClarity_Native.py is missing")}</div>`;
    body += `<div class="small muted">SetiAstro's Cosmic Clarity AI4 models (Sharpen, Denoise, Super-resolution, DarkStar, satellite-trail removal), run by the siril-scripts script CosmicClarity_Native.py. An optional path next to the RC-Astro tools: Sharpen, Denoise, Super-resolution and satellite trails are chosen under Integration &amp; options, DarkStar under Stars &gt; Star removal.</div>`;
    return card("cosmic_clarity", "Cosmic Clarity", t, body + note("cosmic_clarity"), t.version ? "script " + t.version : "");
  }

  function productRow(p, canAct) {
    const c = pcls(p);
    const can = canAct && !T.busy;
    const act = p.valid ? `<button class="btn tiny ghost" data-act="deactivate" data-product="${esc(p.key)}" ${can ? "" : "disabled"}>Deactivate</button>`
      : ["available", "deactivated", "required", "limit"].includes(p.status) ? `<button class="btn tiny" data-act="activate" data-product="${esc(p.key)}" ${can ? "" : "disabled"}>Activate</button>` : "";
    return `<tr><td><b>${esc(p.name || p.key)}</b><br><span class="pstat ${c}">${esc(p.status || "")}</span></td>
      <td class="pmsg">${esc(p.message || "")}${p.modelPresent === false ? `<br><span class="muted">model not downloaded</span>` : ""}</td><td>${act}</td></tr>`;
  }

  function rcastroCard(t) {
    t = t || {};
    const prods = t.products || [];
    const email = typeof t.account === "string" ? t.account : t.account?.email;
    const missingModels = prods.some(p => p.valid && p.modelPresent === false);
    let body;
    if (!ok(t)) {
      body = `<div class="tmsg err">${esc(t.error || t.reason || "The RC-Astro command-line tool is not installed in this image (the installer was not provided at build time). BlurXTerminator, NoiseXTerminator and StarXTerminator are not available; GraXpert and Siril are used instead.")}</div>`;
    } else {
      body = kv([["Account", email ? esc(email) : "not registered"], ["Computer", esc(t.computer)], ["CLI", esc(t.version)]]) +
        (t.attention ? `<div class="attention">${esc(t.attention.message || "")}${t.attention.command ? `<br>Run <code>${esc(t.attention.command)}</code> in an interactive terminal.` : ""}</div>` : "") +
        `<table><tbody>${prods.map(p => productRow(p, true)).join("") || `<tr><td class="muted">No product information yet: press Refresh.</td></tr>`}</tbody></table>` +
        `<div class="small muted">Activation code (8 letters, from your RC-Astro account page):</div>
         <div class="codebox"><input type="text" id="rcCode" maxlength="9" placeholder="XXXX-XXXX" value="${esc(T.code)}" autocomplete="off" spellcheck="false">
           <button class="btn small" data-act="license" ${T.busy ? "disabled" : ""}>Activate</button></div>
         <div class="small"><a href="${esc(t.activationUrl || "https://www.rc-astro.com/account")}" target="_blank" rel="noopener">Get a code</a> <span class="muted">(log in, then copy the code it shows)</span></div>` +
        (missingModels ? `<div class="row gap"><button class="btn small" data-act="dl" data-tool="rcastro" ${T.busy ? "disabled" : ""}>Download RC-Astro models</button></div>` : "") +
        `<div class="small muted">A licence activates on a limited number of computers (a container counts as one). Activations are changed only when a job is not running.</div>`;
    }
    return card("rcastro", "RC-Astro", t, body + note("rcastro"), t.version ? "CLI " + t.version : "");
  }

  function gaiaCard(t) {
    t = t || {};
    const gb = v => v ? (v / 1e9).toFixed(2) + " GB" : "0 GB";
    const chunks = (t.chunks || []).length ? (t.chunks || []).join(", ") : "none yet";
    return card("gaia", "Gaia SPCC data (cached per field)", { ok: true }, kv([["Sky chunks cached", esc(`${t.installed || 0} of ${t.total || 48} (${chunks})`)], ["Size", gb(t.bytes)], ["Folder", esc(t.path)]]) +
      `<div class="small muted">Colour calibration (Siril SPCC) uses Gaia DR3 spectra from Zenodo. The first field calibrated in a region fetches that region's chunk of the catalogue (one of 48, 27-626 MB compressed) and keeps it, so later fields there calibrate offline and in seconds. If the fetch fails the run reads Zenodo directly as before.</div>` +
      (t.paused_until ? `<div class="tmsg err">Fetching is paused until ${new Date(t.paused_until * 1000).toLocaleTimeString()} after a failed download (${esc(t.paused_why || "")}); calibration runs online meanwhile.</div>` : "") +
      ((t.installed || 0) ? `<div class="row gap"><button class="btn small" data-act="gaiaclear" ${T.busy ? "disabled" : ""}>Clear the cache</button></div>` : "") +
      note("gaia"), `${t.installed || 0} chunks`);
  }

  function render() {
    const d = T.data; if (!d) return;
    $("#toolsBody").className = "";
    $("#toolsBody").innerHTML = sirilCard(d.siril) + graxpertCard(d.graxpert) + starnetCard(d.starnet) + rcastroCard(d.rcastro) + ccCard(d.cosmic_clarity) + gaiaCard(d.gaia);
    $$("#toolsBody details.tool").forEach(el => el.ontoggle = () => el.open ? T.open.add(el.dataset.tool) : T.open.delete(el.dataset.tool));
    const code = $("#rcCode"); if (code) code.oninput = () => (T.code = code.value);
    const acc = $("#starnetAccept"); if (acc) acc.onchange = () => (T.starnetOk = acc.checked);
    $$("#toolsBody [data-act]").forEach(b => b.onclick = () => action(b.dataset.act, b.dataset));
  }

  const say = (key, text, err = false) => { T.msg[key] = { text, err }; render(); };

  async function licenceCall(path, body, label) {
    T.busy = label; say("rcastro", label + "…");
    try {
      const r = await api(path, { method: "POST", body });
      T.code = "";
      const rep = r.report || {};
      if (rep.ok === false || rep.error) {                // the CLI refused: show why (its attention message, else its error)
        T.msg.rcastro = { text: rep.attention?.message || rep.error || "RC-Astro refused the request", err: true };
        toast(T.msg.rcastro.text, true);
      } else { T.msg.rcastro = null; toast(label + ": done"); }
      if (r.tools) T.data.rcastro = r.tools;
      await load();
    } catch (e) { T.busy = ""; say("rcastro", e.message, true); toast(e.message, true); return; }
    T.busy = ""; render();
  }

  async function action(act, ds) {
    if (T.busy) return;
    if (act === "license") {
      const code = ($("#rcCode").value || "").replace(/[^A-Za-z0-9]/g, "");
      if (code.length !== 8) return say("rcastro", "Enter the 8-letter activation code from your RC-Astro account page.", true);
      return licenceCall("/api/tools/rcastro/license", { code }, "Registering this computer");
    }
    if (act === "activate") return licenceCall("/api/tools/rcastro/activate", { product: ds.product }, "Activating " + ds.product.toUpperCase());
    if (act === "deactivate") {
      if (!confirm(`Free this computer's ${ds.product.toUpperCase()} activation? You can activate it again later.`)) return;
      return licenceCall("/api/tools/rcastro/deactivate", { product: ds.product }, "Deactivating " + ds.product.toUpperCase());
    }
    if (act === "gaiaclear") {
      if (!confirm("Remove the cached Gaia catalogue chunks? They are fetched again on the next calibration in that part of the sky.")) return;
      try { const r = await api("/api/tools/gaia/clear", { method: "POST", body: {} }); toast(`Removed ${r.removed} files`); await load(); } catch (e) { say("gaia", e.message, true); }
      return;
    }
    if (act === "dl") {
      const tool = ds.tool, body = { tool };
      if (tool === "starnet") {
        if (!$("#starnetAccept")?.checked) return say("starnet", "Tick the licence box first.", true);
        body.accept_licence = true;
      }
      try {
        const job = await api("/api/tools/models/download", { method: "POST", body });
        T.busy = tool; say(tool, "Queued: " + (job.message || "waiting"));
        watch(job.id, tool);
      } catch (e) { say(tool, e.message, true); }
    }
  }

  /* a download runs as a queued job (it never overlaps processing): follow it here */
  async function watch(id, tool) {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch { setTimeout(() => watch(id, tool), 3000); return; }
    if (["queued", "running", "paused"].includes(j.state)) {
      T.msg[tool] = { text: `${j.message} (${(j.progress * 100).toFixed(0)} %)`, err: false };
      const el = $(`details[data-tool="${tool}"] .tmsg`); if (el) el.textContent = T.msg[tool].text; else render();
      return setTimeout(() => watch(id, tool), 1500);
    }
    T.busy = "";
    if (j.state === "done") { T.msg[tool] = null; toast("Download finished"); }
    else T.msg[tool] = { text: j.message || j.state, err: j.state === "error" };
    await load();
  }

  function init() {
    $("#toolsRefresh").onclick = load;
    // the plan names the tool each step will use: follow the choices that change it
    document.addEventListener("change", e => {
      const t = e.target;
      if (["sp-deconv_method", "sp-denoise_method"].includes(t.id) || ["star_removal", "cc_sharpen", "cc_denoise"].includes(t.dataset?.key)) renderPlan();
    });
    load();
  }
  return { init, load, renderPlan, plan };
})();
