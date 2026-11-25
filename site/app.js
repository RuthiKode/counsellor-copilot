// Counsellor Copilot website: recomputes the call plan in the browser.
// Same arithmetic as src/economics.py: EV per call = p × price × margin − cost;
// call in score order while EV > 0, capped at counsellors × calls/day × days.
(async function () {
  const D = await (await fetch("data/site.json")).json();
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const fmt = (x, d = 0) => x.toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const lakh = (x) => `₹${fmt(x / 1e5, 1)} lakh`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const COLORS = { Hot: "#E09F3E", Warm: "#2A7F7A", Cold: "#94A3B8" };

  // Leads arrive in score order; Do Not Call leads are never on a call list.
  const leads = D.leads.filter((l) => !l.dnc);
  const N = leads.length;
  const cum = new Float64Array(N + 1);
  leads.forEach((l, i) => (cum[i + 1] = cum[i] + l.y));
  const total = cum[N], rate = total / N;

  // ------------------------------------------------------------ planner state
  const d = D.defaults;
  const defaults = { counsellors: d.counsellors, cpd: d.calls_per_counsellor_per_day, days: d.days_to_work_batch,
    price: d.course_price, margin: d.gross_margin, cost: d.cost_per_call };
  const show = { counsellors: (v) => v, cpd: (v) => v, days: (v) => v,
    price: (v) => `₹${fmt(v)}`, margin: (v) => pct(v), cost: (v) => `₹${fmt(v)}` };
  const S = { ...defaults };
  const ui = { band: "all", onlyBatch: true, page: 0, q: "", sel: 0, chart: "reach" };

  function plan() {
    const value = S.price * S.margin;
    const breakEven = S.cost / value;
    const cap = S.counsellors * S.cpd * S.days;
    let nEV = 0;
    while (nEV < N && leads[nEV].p * value - S.cost > 0) nEV++;
    const nCut = Math.min(nEV, cap);
    const enrol = cum[nCut];
    const rnd = nCut * rate;
    const extra = S.cpd * S.days;
    const nextEnrol = cum[Math.min(nCut + extra, N)] - enrol;
    return { value, breakEven, cap, nEV, nCut, enrol, rnd, capture: enrol / total,
      cpe: enrol ? nCut / enrol : Infinity, profit: enrol * value - nCut * S.cost,
      rprofit: rnd * value - nCut * S.cost, limitedBy: nCut < nEV ? "capacity" : "breakeven",
      extra, nextEnrol, nextProfit: nextEnrol * value - Math.min(extra, N - nCut) * S.cost };
  }

  // ------------------------------------------------------------ hero + figures
  function renderHero(P) {
    $("#hero-line").innerHTML = `This batch, call the top <mark>${fmt(P.nCut)}</mark> of ${fmt(N)} leads and stop there.`;
    const why = P.limitedBy === "capacity"
      ? "That is all your team can reach in the time you've set."
      : `Past that point a call costs more than it is likely to earn.`;
    $("#hero-sub").innerHTML = `Called in score order, those leads held <b>${pct(P.capture)} of all enrolments</b>, about
      <b>${fmt(P.enrol - P.rnd)} more</b> than the same number of calls in random order. ${why}`;
    $("#figs").innerHTML = [
      [fmt(P.enrol), "enrolments reached", `random order: ${fmt(P.rnd)}`],
      [P.cpe === Infinity ? "–" : fmt(P.cpe, 2), "calls per enrolment", `random order: ${fmt(1 / rate, 2)}`],
      [lakh(P.profit), "projected profit", `${lakh(P.profit - P.rprofit)} more than random`],
      [pct(P.breakEven, 1), "break-even chance to enrol", `₹${fmt(S.cost)} call vs ₹${fmt(P.value)} enrolment`],
    ].map(([v, k, c]) => `<div><dt>${v}</dt><dd class="k">${k}</dd><dd class="c">${c}</dd></div>`).join("");
  }

  // ------------------------------------------------------------ chart (hand-drawn SVG)
  function renderChart(P) {
    const W = 1100, H = 420, m = { l: 64, r: 24, t: 24, b: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const step = Math.max(1, Math.floor(N / 300));
    const idx = []; for (let i = 0; i <= N; i += step) idx.push(i); if (idx[idx.length - 1] !== N) idx.push(N);
    const isReach = ui.chart === "reach";
    const yOf = (i) => isReach ? cum[i] / total : (cum[i] * P.value - i * S.cost);
    const yRand = (i) => isReach ? i / N : (i * rate * P.value - i * S.cost);
    const ys = idx.map(yOf).concat(idx.map(yRand));
    const yMin = isReach ? 0 : Math.min(0, ...ys), yMax = isReach ? 1 : Math.max(...ys) * 1.06;
    const X = (i) => m.l + (i / N) * iw, Y = (v) => m.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
    const path = (f) => idx.map((i, k) => `${k ? "L" : "M"}${X(i).toFixed(1)},${Y(f(i)).toFixed(1)}`).join("");
    const ticksY = isReach ? [0, .2, .4, .6, .8, 1] : niceTicks(yMin, yMax);
    const fy = (v) => isReach ? pct(v) : `₹${fmt(v / 1e5)}L`;
    const cx = X(P.nCut), cy = Y(yOf(P.nCut));
    const label = isReach ? `Stop at ${fmt(P.nCut)} calls: ${pct(P.capture)} of enrolments`
                          : `Stop at ${fmt(P.nCut)} calls: ${lakh(P.profit)}`;
    const lx = cx + 14 + 330 > W ? cx - 14 : cx + 14, anchor = lx < cx ? "end" : "start";
    $("#chart").setAttribute("aria-label", label);
    $("#chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}">
      <rect x="${m.l}" y="${m.t}" width="${Math.max(0, cx - m.l)}" height="${ih}" fill="#F7E4C3" opacity=".7"/>
      <g class="grid">${ticksY.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join("")}</g>
      <g class="axis">${ticksY.map((t) => `<text x="${m.l - 10}" y="${Y(t) + 4}" text-anchor="end">${fy(t)}</text>`).join("")}
        ${[0, .2, .4, .6, .8, 1].map((t) => `<text x="${X(t * N)}" y="${H - 22}" text-anchor="middle">${pct(t)}</text>`).join("")}
        <text x="${m.l + iw / 2}" y="${H - 2}" text-anchor="middle">Share of leads called, highest score first</text></g>
      <path d="${path(yRand)}" fill="none" stroke="#94A3B8" stroke-width="2" stroke-dasharray="6 6"/>
      <path d="${path(yOf)}" fill="none" stroke="#2A7F7A" stroke-width="3.5" stroke-linejoin="round"/>
      ${P.nEV !== P.nCut ? `<line x1="${X(P.nEV)}" x2="${X(P.nEV)}" y1="${m.t}" y2="${m.t + ih}" stroke="#5B6577" stroke-dasharray="2 4"/>
        <text x="${X(P.nEV) - 8}" y="${m.t + ih - 10}" text-anchor="end" fill="#5B6577" font-size="13">break-even would allow ${fmt(P.nEV)}</text>` : ""}
      <circle cx="${cx}" cy="${cy}" r="8" fill="#E09F3E" stroke="#14213D" stroke-width="2"/>
      <text class="lbl" x="${lx}" y="${cy - 12}" text-anchor="${anchor}">${label}</text>
    </svg>`;
    $("#add-counsellor").innerHTML = P.nCut >= N
      ? "Your team can already reach every lead in this batch."
      : `<b>Thinking of adding a counsellor?</b> One more makes ${fmt(P.extra)} calls per batch. The next leads below
         today's cutoff would bring about <b>${fmt(P.nextEnrol)} more enrolments</b> (${lakh(P.nextProfit)} before salary).`;
  }
  function niceTicks(a, b) {
    const span = b - a, raw = span / 5, mag = 10 ** Math.floor(Math.log10(raw));
    const st = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((k) => span / k <= 6);
    const out = []; for (let v = Math.ceil(a / st) * st; v <= b; v += st) out.push(v); return out;
  }

  // ------------------------------------------------------------ call list
  const PAGE = 20;
  function filtered(P) {
    return leads.map((l, i) => ({ ...l, rank: i + 1 }))
      .filter((l) => (!ui.onlyBatch || l.rank <= P.nCut) && (ui.band === "all" || l.b === ui.band)
        && (!ui.q || String(l.n).includes(ui.q)));
  }
  function renderSplit(P) {
    const c = { Hot: 0, Warm: 0, Cold: 0 };
    for (let i = 0; i < P.nCut; i++) c[leads[i].b]++;
    const parts = [...Object.entries(c).map(([b, k]) => [b, k, `${b} ${fmt(k)}`]), ["rest", N - P.nCut, `Not called ${fmt(N - P.nCut)}`]];
    $("#split").innerHTML = parts.filter(([, k]) => k > 0).map(([cls, k, t]) => `<div class="${cls}" style="width:${(k / N) * 100}%" title="${t}">${k / N > .07 ? t : ""}</div>`).join("");
    const cu = D.cutoffs;
    $("#split-legend").innerHTML = `<span><i style="background:${COLORS.Hot}"></i>Hot: score ${fmt(cu.hot)}+, the top 20%</span>
      <span><i style="background:${COLORS.Warm}"></i>Warm: ${fmt(cu.warm)} to ${fmt(cu.hot)}, the next 30%</span>
      <span><i style="background:${COLORS.Cold}"></i>Cold: below ${fmt(cu.warm)}</span>`;
  }
  function renderTable(P) {
    const rows = filtered(P);
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    ui.page = Math.min(ui.page, pages - 1);
    const slice = rows.slice(ui.page * PAGE, ui.page * PAGE + PAGE);
    $("#leads tbody").innerHTML = slice.length ? slice.map((l) => `
      <tr data-rank="${l.rank}" tabindex="0" class="${l.rank - 1 === ui.sel ? "sel" : ""} ${l.rank > P.nCut ? "out" : ""}">
        <td class="num">${l.rank}</td><td class="num">${l.n}</td>
        <td><div class="score"><span class="num">${fmt(l.s)}</span><div class="bar"><span style="width:${l.s}%"></span></div></div></td>
        <td><span class="pill ${l.b}">${l.b}</span></td><td>${esc(l.g)}</td>
        <td class="why">${l.d.map((x) => `${esc(x.l)}: ${esc(x.v)} <span class="${x.up ? "up" : "down"}">${x.up ? "↑" : "↓"}</span>`).join("; ")}</td>
      </tr>`).join("") : `<tr><td colspan="6">No leads match. Clear the search or pick another band.</td></tr>`;
    $("#pager").innerHTML = `<span>${fmt(rows.length)} leads, page ${ui.page + 1} of ${pages}</span>
      <button class="btn small" data-p="-1" ${ui.page === 0 ? "disabled" : ""} aria-label="Previous page">‹</button>
      <button class="btn small" data-p="1" ${ui.page >= pages - 1 ? "disabled" : ""} aria-label="Next page">›</button>`;
  }

  // ------------------------------------------------------------ slip
  function renderSlip() {
    const l = leads[ui.sel];
    $("#slip").className = `slip ${l.b}`;
    $("#slip").innerHTML = `<header><h3>Lead ${l.n}</h3><span class="rank">#${ui.sel + 1} in the queue</span></header>
      <p class="meta">Score ${fmt(l.s)} out of 100, ${l.b}, ${esc(l.g)}</p>
      <dl><dt>Why</dt><dd>${esc(l.why)}</dd><dt>Objection</dt><dd>${esc(l.obj)}</dd>
          <dt>Opener</dt><dd class="open">${esc(l.open)}</dd></dl>
      <div class="tags">${Object.entries(l.facts).map(([k, v]) => `<span>${esc(k)}: ${esc(v)}</span>`).join("")}</div>`;
    $("#prev").disabled = ui.sel === 0;
  }

  // ------------------------------------------------------------ static sections
  function renderGroups() {
    const base = D.personas.reduce((a, p) => a + p.conversions, 0) / D.personas.reduce((a, p) => a + p.leads, 0);
    $("#group-list").innerHTML = D.personas.map((p) => {
      const col = p.lift >= 1.5 ? COLORS.Hot : p.lift >= .8 ? COLORS.Warm : COLORS.Cold;
      return `<div class="group"><div><h3>${esc(p.persona)}</h3><div class="size">${pct(p.share_of_leads)} of leads,
          ${pct(p.share_of_conversions)} of enrolments</div></div>
        <div class="rate"><div class="track"><span style="width:${p.conv_rate * 100}%;background:${col}"></span>
          <i style="left:${base * 100}%"></i></div><b>${pct(p.conv_rate)}</b></div>
        <div class="share">${p.lift.toFixed(1)}× the average<br><span class="size">about ${fmt(p.avg_minutes_on_site)} min on the site</span></div>
        <div class="play">${esc(p.playbook)}</div></div>`;
    }).join("") + `<p class="avg-note">The black tick marks the ${pct(base)} average enrol rate. Rates are from past leads the model never trained on.</p>`;
  }
  function renderMethod() {
    const L = D.leakage, keys = Object.keys(L);
    const honest = keys.find((k) => k.startsWith("Honest")), leaky = keys.find((k) => k.startsWith("Leaky"));
    const row = (name, v, f, col) => `<div class="row"><span>${name}</span><div class="bar"><span style="width:${v * 100}%;background:${col}"></span></div><span class="val">${f}</span></div>`;
    $("#leak-bars").innerHTML = `<div class="lb"><h4>How well it separates enrolled from not (AUC)</h4>
        ${row("Honest model", L[honest].auc, L[honest].auc.toFixed(3), COLORS.Warm)}${row("With call notes", L[leaky].auc, L[leaky].auc.toFixed(3), COLORS.Cold)}
        <h4>Enrolled share of the top 20% of leads</h4>
        ${row("Honest model", L[honest].p20, pct(L[honest].p20, 1), COLORS.Warm)}${row("With call notes", L[leaky].p20, pct(L[leaky].p20, 1), COLORS.Cold)}</div>
      <p>The honest model is the one this page uses. Its top-20% figure is what a team could expect on brand-new leads.</p>`;
    $("#rec-list").innerHTML = D.recommendations.map((r) => `<div class="rec"><div class="owner">${esc(r.owner)}</div>
      <div class="act">${esc(r.action)}</div><div class="ev">${esc(r.evidence)} ${esc(r.effect)}</div></div>`).join("");
  }

  // ------------------------------------------------------------ wiring
  function update() {
    for (const k of Object.keys(S)) $(`output[data-for=${k}]`).textContent = show[k](S[k]);
    const P = plan();
    renderHero(P); renderChart(P); renderSplit(P); renderTable(P);
  }
  for (const k of Object.keys(S)) {
    const el = $(`#${k}`); el.value = S[k];
    el.addEventListener("input", () => { S[k] = +el.value; update(); });
  }
  $("#reset").addEventListener("click", () => { Object.assign(S, defaults); for (const k of Object.keys(S)) $(`#${k}`).value = S[k]; update(); });
  $$(".toggle button").forEach((b) => b.addEventListener("click", () => {
    ui.chart = b.dataset.chart; $$(".toggle button").forEach((x) => x.setAttribute("aria-selected", x === b)); renderChart(plan());
  }));
  $$(".chip").forEach((c) => c.addEventListener("click", () => {
    ui.band = c.dataset.band; ui.page = 0; $$(".chip").forEach((x) => x.classList.toggle("on", x === c)); renderTable(plan());
  }));
  $("#only-batch").addEventListener("change", (e) => { ui.onlyBatch = e.target.checked; ui.page = 0; renderTable(plan()); });
  $("#search").addEventListener("input", (e) => { ui.q = e.target.value.trim(); ui.page = 0; renderTable(plan()); });
  $("#pager").addEventListener("click", (e) => { const b = e.target.closest("button[data-p]"); if (!b) return; ui.page += +b.dataset.p; renderTable(plan()); });
  const pick = (tr) => { ui.sel = +tr.dataset.rank - 1; renderSlip(); renderTable(plan()); $("#brief").scrollIntoView(); };
  $("#leads tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-rank]"); if (tr) pick(tr); });
  $("#leads tbody").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-rank]"); if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pick(tr); } });
  $("#next").addEventListener("click", () => { ui.sel = Math.min(N - 1, ui.sel + 1); renderSlip(); });
  $("#prev").addEventListener("click", () => { ui.sel = Math.max(0, ui.sel - 1); renderSlip(); });
  $("#download").addEventListener("click", () => {
    const P = plan();
    const lines = ["rank,in_batch,lead_number,score,band,lead_group,reasons"].concat(leads.map((l, i) =>
      [i + 1, i < P.nCut, l.n, l.s, l.b, `"${l.g}"`, `"${l.d.map((x) => `${x.l}: ${x.v} ${x.up ? "up" : "down"}`).join("; ")}"`].join(",")));
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })), download: "call_list.csv" });
    a.click(); URL.revokeObjectURL(a.href);
  });
  // highlight the section in view
  const navLinks = $$(".nav nav a");
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main section").forEach((s) => io.observe(s));

  renderGroups(); renderMethod(); renderSlip(); update();
})();
