import { useState, useMemo, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════
//  CONSTANTS
// ═══════════════════════════════════════════════════════════════════

const EPS       = 0.1;   // BSC flip probability
const GRID      = 4;
const MAX_Q     = 10;    // max questions per game
const THRESHOLD = 0.95;  // convergence: max weight > this
const N_SIM     = 5000;  // games per (strategy × true-hyp)
const THRESHOLD_DEFAULT = 0.95;

// Four hypotheses: a single length-2 ship on a 4×4 grid
const HYPS = [
  { id:0, label:"H1", desc:"Horiz · row 0 · cols 0–1", cells:[[0,0],[0,1]], color:"#3b82f6", prior:0.45 },
  { id:1, label:"H2", desc:"Horiz · row 1 · cols 0–1", cells:[[1,0],[1,1]], color:"#ef4444", prior:0.45 },
  { id:2, label:"H3", desc:"Vert  · rows 0–1 · col 2", cells:[[0,2],[1,2]], color:"#22c55e", prior:0.05 },
  { id:3, label:"H4", desc:"Vert  · rows 1–2 · col 2", cells:[[1,2],[2,2]], color:"#f97316", prior:0.05 },
];
const INIT_W = HYPS.map(h => h.prior);

// ═══════════════════════════════════════════════════════════════════
//  MATH
// ═══════════════════════════════════════════════════════════════════

const lg2  = x => x <= 0 ? 0 : Math.log2(x);
const hb   = p => (p <= 1e-12 || p >= 1-1e-12) ? 0 : -p*lg2(p) - (1-p)*lg2(1-p);
const Hsh  = ws => ws.reduce((s, w) => s + (w > 1e-12 ? -w*lg2(w) : 0), 0);

// True (noiseless) answer for hypothesis h and question q
const fq = (h, q) => {
  if (q.t === "cell") return h.cells.some(([r,c]) => r===q.r && c===q.c) ? 1 : 0;
  if (q.t === "row")  return h.cells.some(([r,c]) => r===q.r)            ? 1 : 0;
  /* col */           return h.cells.some(([r,c]) => c===q.c)            ? 1 : 0;
};

// P(Ã=1 | h, q) under BSC(ε)
const pHYes = (h, q) => EPS + (1 - 2*EPS) * fq(h, q);

// Marginal P(Ã=1 | q, belief ws)
const margYes = (ws, q) => HYPS.reduce((s, h, i) => s + ws[i]*pHYes(h,q), 0);

// EIG = H_b(p_marginal) − H_b(ε)
//   Since p_s(Yes) ∈ {ε, 1−ε}, H(Ã|S) = H_b(ε) regardless of belief.
const computeEIG = (ws, q) => hb(margYes(ws, q)) - hb(EPS);

// E[log LR] = Σ_s π_s · KL(p_s ‖ p_{−s})   (natural log; nats)
const computeELLR = (ws, q) => {
  const py = margYes(ws, q), pn = 1 - py;
  return HYPS.reduce((tot, h, i) => {
    const pi = ws[i]; if (pi < 1e-12) return tot;
    const sy = pHYes(h,q), sn = 1-sy, d = 1-pi; if (d < 1e-12) return tot;
    const my = (py - pi*sy)/d,  mn = (pn - pi*sn)/d;
    let kl = 0;
    if (sy > 1e-12 && my > 1e-12) kl += sy * Math.log(sy/my);
    if (sn > 1e-12 && mn > 1e-12) kl += sn * Math.log(sn/mn);
    return tot + pi*kl;
  }, 0);
};

// Bayesian update after observing ans ∈ {0,1}
const updateW = (ws, q, ans) => {
  const raw = ws.map((pi, i) => { const py = pHYes(HYPS[i], q); return pi*(ans ? py : 1-py); });
  const Z = raw.reduce((a,b) => a+b, 0);
  return raw.map(w => w/Z);
};

// ═══════════════════════════════════════════════════════════════════
//  QUESTION CATALOGUE  16 cell + 4 row + 4 col = 24 total
// ═══════════════════════════════════════════════════════════════════

const QUESTIONS = (() => {
  const qs = [];
  for (let r=0; r<GRID; r++)
    for (let c=0; c<GRID; c++)
      qs.push({ id:`cell_${r}_${c}`, t:"cell", r, c,
                label:`Cell (${r},${c})`, group:"Cell" });
  for (let r=0; r<GRID; r++)
    qs.push({ id:`row_${r}`, t:"row", r,
              label:`Row ${r}`, group:"Row" });
  for (let c=0; c<GRID; c++)
    qs.push({ id:`col_${c}`, t:"col", c,
              label:`Col ${c}`, group:"Col" });
  return qs;
})();

// ═══════════════════════════════════════════════════════════════════
//  SIMULATION
// ═══════════════════════════════════════════════════════════════════

const runGame = (strategy, trueHypIdx, threshold) => {
  let ws = [...INIT_W], nq = 0;
  while (nq < MAX_Q && Math.max(...ws) < threshold) {
    let bestQ = QUESTIONS[0], bestS = -Infinity;
    for (const q of QUESTIONS) {
      const s = strategy === "eig" ? computeEIG(ws, q) : computeELLR(ws, q);
      if (s > bestS) { bestS = s; bestQ = q; }
    }
    const py  = pHYes(HYPS[trueHypIdx], bestQ);
    const ans = Math.random() < py ? 1 : 0;
    ws = updateW(ws, bestQ, ans);
    nq++;
  }
  return { nq, correct: ws.indexOf(Math.max(...ws)) === trueHypIdx };
};

// ═══════════════════════════════════════════════════════════════════
//  STYLES  (analytical / instrument-panel aesthetic)
// ═══════════════════════════════════════════════════════════════════

const FONT = "'IBM Plex Mono', 'Fira Code', monospace";

const S = {
  page:    { fontFamily:FONT, fontSize:12, padding:16, maxWidth:1140,
             margin:"0 auto", color:"#0f172a", background:"#f8fafc", minHeight:"100vh" },
  card:    { border:"1px solid #cbd5e1", borderRadius:6, padding:12,
             marginBottom:10, background:"#fff" },
  row:     { display:"flex", gap:10, flexWrap:"wrap", marginBottom:10 },
  h2:      { margin:"0 0 8px", fontSize:14, fontWeight:700, letterSpacing:"0.02em" },
  h3:      { margin:"0 0 6px", fontSize:12, fontWeight:700, color:"#475569",
             textTransform:"uppercase", letterSpacing:"0.06em" },
  th:      { padding:"5px 8px", textAlign:"left", background:"#1e293b", color:"#e2e8f0",
             fontSize:11, fontWeight:600, whiteSpace:"nowrap", letterSpacing:"0.04em" },
  td:      { padding:"4px 8px", borderBottom:"1px solid #f1f5f9", fontSize:11,
             verticalAlign:"middle" },
  btn:     (bg, fg="#fff") => ({
             display:"inline-block", padding:"5px 12px", background:bg, color:fg,
             border:"none", borderRadius:4, cursor:"pointer", fontSize:11,
             fontFamily:FONT, fontWeight:600, letterSpacing:"0.04em" }),
  pill:    (bg) => ({
             display:"inline-block", padding:"1px 7px", borderRadius:10,
             background:bg+"22", color:bg, fontSize:10, fontWeight:700,
             border:`1px solid ${bg}55`, letterSpacing:"0.03em" }),
  mono:    { fontFamily:FONT },
};

// ═══════════════════════════════════════════════════════════════════
//  COMPONENT
// ═══════════════════════════════════════════════════════════════════

export default function App() {
  const [weights, setWeights] = useState(INIT_W);
  const [trueIdx, setTrueIdx] = useState(0);
  const [history, setHistory] = useState([]);
  const [simRes,  setSimRes]  = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [sortKey, setSortKey] = useState("eig");
  const [hover,   setHover]   = useState(null);
  const [running,   setRunning]   = useState(false);
  const [threshold, setThreshold] = useState(THRESHOLD_DEFAULT);
  const [priorWeighted, setPriorWeighted] = useState(false);

  // ── Score every question at current belief ──────────────────────
  const scored = useMemo(() =>
    QUESTIONS.map(q => ({
      ...q,
      eigScore:  computeEIG(weights, q),
      ellrScore: computeELLR(weights, q),
      pYes:      margYes(weights, q),
    })),
  [weights]);

  const eigBest  = useMemo(() => [...scored].sort((a,b) => b.eigScore  - a.eigScore )[0], [scored]);
  const ellrBest = useMemo(() => [...scored].sort((a,b) => b.ellrScore - a.ellrScore)[0], [scored]);

  const eigRanks  = useMemo(() => {
    const sorted = [...scored].sort((a,b) => b.eigScore  - a.eigScore);
    const m = {}; sorted.forEach((q,i) => m[q.id] = i+1); return m;
  }, [scored]);
  const ellrRanks = useMemo(() => {
    const sorted = [...scored].sort((a,b) => b.ellrScore - a.ellrScore);
    const m = {}; sorted.forEach((q,i) => m[q.id] = i+1); return m;
  }, [scored]);

  // ── Filtered / sorted display table ────────────────────────────
  const displayed = useMemo(() => {
    const qs = showAll ? scored : scored.filter(q => q.eigScore > 0.001 || q.ellrScore > 0.001);
    return [...qs].sort((a,b) => sortKey === "eig" ? b.eigScore-a.eigScore : b.ellrScore-a.ellrScore);
  }, [scored, showAll, sortKey]);

  // ── Ask a question ───────────────────────────────────────────────
  const ask = useCallback((q, strat) => {
    const py  = pHYes(HYPS[trueIdx], q);
    const ans = Math.random() < py ? 1 : 0;
    const wNext = updateW(weights, q, ans);
    setHistory(h => [...h, {
      q, ans, strat,
      wBefore: [...weights], wAfter: wNext,
      eig:  computeEIG(weights, q),
      ellr: computeELLR(weights, q),
    }]);
    setWeights(wNext);
  }, [weights, trueIdx]);

  const reset = () => { setWeights(INIT_W); setHistory([]); setSimRes(null); };

  // ── Bulk simulation ──────────────────────────────────────────────
  const runSim = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const res = {};
      for (const s of ["eig","ellr"])
        res[s] = { byHyp: HYPS.map(() => ({nq:0, ok:0, n:0})), total:{nq:0,ok:0,n:0} };

      for (let g=0; g<N_SIM; g++) {
        for (const s of ["eig","ellr"]) {
          for (let hi=0; hi<HYPS.length; hi++) {
            const r = runGame(s, hi, threshold);
            res[s].byHyp[hi].nq += r.nq;
            res[s].byHyp[hi].ok += r.correct ? 1 : 0;
            res[s].byHyp[hi].n  += 1;
            res[s].total.nq += r.nq;
            res[s].total.ok += r.correct ? 1 : 0;
            res[s].total.n  += 1;
          }
        }
      }
      setSimRes(res);
      setRunning(false);
    }, 10);
  }, [threshold]);

  // ── Grid occupancy heatmap ───────────────────────────────────────
  const occ = useMemo(() => {
    const g = Array.from({length:GRID}, () => Array(GRID).fill(0));
    HYPS.forEach((h, i) => h.cells.forEach(([r,c]) => g[r][c] += weights[i]));
    return g;
  }, [weights]);

  // Highlight cells affected by hovered question
  const isHighlit = (r, c) => {
    if (!hover) return false;
    const q = QUESTIONS.find(x => x.id === hover);
    if (!q) return false;
    if (q.t === "cell") return q.r === r && q.c === c;
    if (q.t === "row")  return q.r === r;
    return q.c === c;
  };

  const disagree = eigBest.id !== ellrBest.id;
  const maxW     = Math.max(...weights);
  const entH     = Hsh(weights);

  return (
    <div style={S.page}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <div style={{ textAlign:"center", marginBottom:12, borderBottom:"2px solid #1e293b", paddingBottom:10 }}>
        <div style={{ fontSize:18, fontWeight:700, letterSpacing:"0.02em", color:"#0f172a" }}>
          EIG vs E[log LR] — Question Selection
        </div>
        <div style={{ fontSize:11, color:"#64748b", marginTop:4 }}>
          4×4 grid · 1 ship (length 2) · 4 hypotheses · prior [H1=0.45, H2=0.45, H3=0.05, H4=0.05] · 24 candidate questions · BSC noise ε={EPS}
        </div>
      </div>

      <div style={S.row}>

        {/* ── LEFT: Grid + Hypotheses ────────────────────────────── */}
        <div style={{ ...S.card, minWidth:290 }}>
          <div style={S.h3}>Ship probability per cell</div>
          <div style={{ display:"grid", gridTemplateColumns:"22px repeat(4, 52px)", gap:2, marginBottom:12 }}>
            <div/>
            {[0,1,2,3].map(c =>
              <div key={c} style={{ textAlign:"center", fontWeight:700, fontSize:10, color:"#94a3b8", padding:"2px 0" }}>
                C{c}
              </div>
            )}
            {Array.from({length:GRID}, (_,r) => [
              <div key={`rl${r}`} style={{ fontWeight:700, fontSize:10, color:"#94a3b8", display:"flex", alignItems:"center" }}>
                R{r}
              </div>,
              ...Array.from({length:GRID}, (_,c) => {
                const p  = occ[r][c];
                const hi = isHighlit(r, c);
                const alpha = p > 0 ? 0.12 + 0.75*p : 0;

                // Ground-truth rectangle: color exterior edges, suppress shared interior edge
                const gtCells = HYPS[trueIdx].cells;
                const gtColor = HYPS[trueIdx].color;
                const inGT = gtCells.some(([gr,gc]) => gr===r && gc===c);
                const bw = "2.5px";
                let borderTop, borderRight, borderBottom, borderLeft;
                if (inGT) {
                  const other = gtCells.find(([gr,gc]) => !(gr===r && gc===c));
                  const [or, oc] = other;
                  const isHoriz = or === r;   // other cell is in same row
                  const solid   = `${bw} solid ${gtColor}`;
                  const inner   = `1px solid #e2e8f0`;
                  borderTop    = solid;
                  borderRight  = isHoriz && oc === c+1 ? inner : solid;
                  borderBottom = solid;
                  borderLeft   = isHoriz && oc === c-1 ? inner : solid;
                  if (!isHoriz) {
                    borderBottom = or === r+1 ? inner : solid;
                    borderTop    = or === r-1 ? inner : solid;
                    borderRight  = solid;
                    borderLeft   = solid;
                  }
                } else {
                  const plain = hi ? `2px solid #f59e0b` : "1px solid #e2e8f0";
                  borderTop = borderRight = borderBottom = borderLeft = plain;
                }

                return (
                  <div key={`g${r}${c}`} style={{
                    width:52, height:44,
                    background: p > 0 ? `rgba(59,130,246,${alpha})` : "#f8fafc",
                    borderTop, borderRight, borderBottom, borderLeft,
                    borderRadius:4, display:"flex", flexDirection:"column",
                    alignItems:"center", justifyContent:"center",
                    color: p > 0.45 ? "#fff" : "#334155",
                    transition:"background 0.3s",
                  }}>
                    <b style={{ fontSize:11 }}>{p > 0 ? `${(p*100).toFixed(0)}%` : "–"}</b>
                    <span style={{ fontSize:9, opacity:0.55 }}>({r},{c})</span>
                  </div>
                );
              })
            ])}
          </div>

          <div style={S.h3}>Hypotheses — click to set ground truth</div>
          {HYPS.map((h, i) => (
            <div key={h.id} onClick={() => setTrueIdx(i)} style={{
              display:"flex", alignItems:"center", gap:6, marginBottom:5,
              padding:"5px 8px", borderRadius:5, cursor:"pointer",
              border:`${trueIdx===i ? 2 : 1}px solid ${trueIdx===i ? h.color : "#e2e8f0"}`,
              background: trueIdx===i ? h.color+"0d" : "#fafafa",
            }}>
              <span style={S.pill(h.color)}>{h.label}</span>
              <span style={{ flex:1, fontSize:11, color:"#475569" }}>{h.desc}</span>
              <div style={{ width:64, height:8, background:"#e2e8f0", borderRadius:4, overflow:"hidden" }}>
                <div style={{ width:`${weights[i]*100}%`, height:"100%", background:h.color, transition:"width 0.4s" }}/>
              </div>
              <span style={{ minWidth:36, textAlign:"right", fontWeight:700, color:h.color, fontSize:12 }}>
                {(weights[i]*100).toFixed(1)}%
              </span>
            </div>
          ))}

          <div style={{ fontSize:11, color:"#64748b", marginTop:6, display:"flex", justifyContent:"space-between" }}>
            <span>Entropy: <b style={{ color:"#0f172a" }}>{entH.toFixed(3)} bits</b></span>
            {maxW >= threshold &&
              <span style={{ color:"#16a34a", fontWeight:700 }}>✓ CONVERGED</span>}
          </div>
        </div>

        {/* ── RIGHT: Divergence banner + controls + history ───────── */}
        <div style={{ flex:1, minWidth:320 }}>

          {/* Divergence panel */}
          <div style={{
            ...S.card,
            borderWidth:2,
            borderColor: disagree ? "#f59e0b" : "#22c55e",
            background:  disagree ? "#fffbeb" : "#f0fdf4",
          }}>
            <div style={{ ...S.h2, color: disagree ? "#78350f" : "#14532d", display:"flex", alignItems:"center", gap:8 }}>
              {disagree
                ? "⚠  CRITERIA DISAGREE — different questions selected at this belief state"
                : "✓  Both criteria agree on the same question"}
            </div>
            <div style={{ display:"flex", gap:10 }}>
              {[["eig","EIG","#3b82f6",eigBest],["ellr","E[log LR]","#ef4444",ellrBest]].map(([key,lbl,col,best]) => (
                <div key={key} style={{
                  flex:1, padding:"10px 12px",
                  background: col+"0f", borderRadius:5, border:`2px solid ${col}`,
                }}>
                  <div style={{ color:col, fontWeight:700, fontSize:12, marginBottom:4 }}>
                    {lbl} picks:
                  </div>
                  <div style={{ fontWeight:700, fontSize:14, marginBottom:3 }}>{best.label}</div>
                  <div style={{ fontSize:11, color:"#475569", lineHeight:1.8 }}>
                    <span style={{ display:"block" }}>EIG = <b>{best.eigScore.toFixed(4)}</b> bits</span>
                    <span style={{ display:"block" }}>E[log LR] = <b>{best.ellrScore.toFixed(4)}</b></span>
                    <span style={{ display:"block" }}>P(Yes) = <b>{best.pYes.toFixed(3)}</b></span>
                  </div>
                </div>
              ))}
            </div>
            {disagree && (
              <div style={{ marginTop:8, fontSize:11, color:"#78350f", lineHeight:1.6 }}>
                <b>EIG</b> prefers the question with P(Yes)≈0.5 (maximising global entropy reduction).<br/>
                <b>E[log LR]</b> prefers cell questions that perfectly isolate the dominant hypothesis (weight 0.45).
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ ...S.card, display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
            <button style={S.btn("#3b82f6")} onClick={() => ask(eigBest, "eig")}>
              ▶ Ask EIG-best
            </button>
            <button style={S.btn("#ef4444")} onClick={() => ask(ellrBest, "ellr")}>
              ▶ Ask E[logLR]-best
            </button>
            <button style={S.btn("#64748b")} onClick={reset}>↺ Reset</button>
          <div style={{ display:"flex", alignItems:"center", gap:7, fontSize:11, color:"#475569" }}>
            <label style={{ whiteSpace:"nowrap" }}>
              Convergence threshold:
            </label>
            <input
              type="range" min={0.60} max={0.99} step={0.01}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{ width:100, accentColor:"#3b82f6" }}
            />
            <span style={{
              minWidth:36, fontWeight:700, color:"#0f172a",
              background:"#f1f5f9", padding:"1px 6px", borderRadius:4,
              border:"1px solid #cbd5e1",
            }}>
              {(threshold*100).toFixed(0)}%
            </span>
          </div>
          <button style={S.btn(running ? "#94a3b8" : "#16a34a")}
                  onClick={runSim} disabled={running}>
            {running ? "Running…" : `⚙ Simulate ${N_SIM}×4 games`}
          </button>
          </div>

          {/* Question history */}
          {history.length > 0 && (
            <div style={S.card}>
              <div style={S.h3}>Question history</div>
              {history.map((step, i) => (
                <div key={i} style={{
                  display:"flex", gap:6, alignItems:"center",
                  padding:"4px 7px", marginBottom:3, borderRadius:4, fontSize:11,
                  background: step.strat==="eig" ? "#eff6ff" : "#fff1f2",
                  border:`1px solid ${step.strat==="eig" ? "#bfdbfe" : "#fecaca"}`,
                }}>
                  <span style={{ color:"#94a3b8", minWidth:20 }}>#{i+1}</span>
                  <span style={S.pill(step.strat==="eig" ? "#3b82f6" : "#ef4444")}>
                    {step.strat.toUpperCase()}
                  </span>
                  <span style={{ fontWeight:700, minWidth:90 }}>{step.q.label}</span>
                  <span>→ <b style={{ color:step.ans ? "#16a34a" : "#ef4444" }}>
                    {step.ans ? "YES" : "NO"}
                  </b></span>
                  <span style={{ color:"#94a3b8" }}>
                    [EIG={step.eig.toFixed(3)}, ELLR={step.ellr.toFixed(3)}]
                  </span>
                  <span style={{ marginLeft:"auto", color:"#64748b", fontSize:10 }}>
                    [{step.wAfter.map(w => `${(w*100).toFixed(0)}%`).join(", ")}]
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════
           QUESTION TABLE
         ══════════════════════════════════════════════════════════ */}
      <div style={S.card}>
        <div style={{ display:"flex", gap:10, alignItems:"center", marginBottom:8, flexWrap:"wrap" }}>
          <div style={S.h2}>All 24 Candidate Questions</div>
          <label style={{ fontSize:11, color:"#475569" }}>
            Sort by:{" "}
            <select value={sortKey} onChange={e => setSortKey(e.target.value)}
                    style={{ fontSize:11, fontFamily:FONT, marginLeft:4 }}>
              <option value="eig">EIG</option>
              <option value="ellr">E[log LR]</option>
            </select>
          </label>
          <label style={{ fontSize:11, color:"#475569" }}>
            <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)}
                   style={{ marginRight:4 }}/>
            Include uninformative (EIG≈0)
          </label>
          <span style={{ fontSize:10, color:"#94a3b8", marginLeft:"auto" }}>
            Hover a row to highlight affected cells in the grid
          </span>
        </div>

        {/* Legend */}
        <div style={{ display:"flex", gap:12, marginBottom:8, fontSize:11, flexWrap:"wrap" }}>
          <span>
            <span style={{ display:"inline-block", width:12, height:12, background:"#eff6ff",
                           border:"2px solid #3b82f6", borderRadius:2, verticalAlign:"middle", marginRight:4 }}/>
            EIG #1 pick
          </span>
          <span>
            <span style={{ display:"inline-block", width:12, height:12, background:"#fff1f2",
                           border:"2px solid #ef4444", borderRadius:2, verticalAlign:"middle", marginRight:4 }}/>
            E[log LR] #1 pick
          </span>
          <span>
            <span style={{ display:"inline-block", width:12, height:12, background:"#dcfce7",
                           border:"2px solid #22c55e", borderRadius:2, verticalAlign:"middle", marginRight:4 }}/>
            Both agree
          </span>
          <span style={{ color:"#94a3b8" }}>
            Δ = |EIG rank − ELLR rank| · large Δ = the two criteria rank this question very differently
          </span>
        </div>

        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse" }}>
            <thead>
              <tr>
                {["Question","Type","P(Yes)","EIG (bits)","EIG rank","E[log LR]","ELLR rank","Rank Δ","Ask"].map(h => (
                  <th key={h} style={S.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayed.map(q => {
                const isEB    = q.id === eigBest.id;
                const isLB    = q.id === ellrBest.id;
                const eRank   = eigRanks[q.id];
                const lRank   = ellrRanks[q.id];
                const delta   = Math.abs(eRank - lRank);
                const rowBg   = isEB && isLB ? "#dcfce7"
                              : isEB ? "#eff6ff"
                              : isLB ? "#fff1f2"
                              : hover === q.id ? "#fefce8"
                              : "white";
                const typeCol = q.t==="cell" ? "#7c3aed" : q.t==="row" ? "#0369a1" : "#0891b2";
                return (
                  <tr key={q.id} style={{ background:rowBg }}
                      onMouseEnter={() => setHover(q.id)}
                      onMouseLeave={() => setHover(null)}>
                    <td style={{ ...S.td, fontWeight: isEB||isLB ? 700 : "normal" }}>
                      {q.label}
                      {isEB && !isLB && <span style={{ ...S.pill("#3b82f6"), marginLeft:5 }}>EIG #1</span>}
                      {isLB && !isEB && <span style={{ ...S.pill("#ef4444"), marginLeft:5 }}>ELLR #1</span>}
                      {isEB && isLB  && <span style={{ ...S.pill("#22c55e"), marginLeft:5 }}>BOTH #1</span>}
                    </td>
                    <td style={S.td}>
                      <span style={S.pill(typeCol)}>{q.group}</span>
                    </td>
                    <td style={S.td}>{q.pYes.toFixed(3)}</td>
                    <td style={{ ...S.td, color:isEB?"#3b82f6":"inherit", fontWeight:isEB?700:"normal" }}>
                      {q.eigScore.toFixed(4)}
                    </td>
                    <td style={{ ...S.td, color:"#64748b" }}>#{eRank}</td>
                    <td style={{ ...S.td, color:isLB?"#ef4444":"inherit", fontWeight:isLB?700:"normal" }}>
                      {q.ellrScore.toFixed(4)}
                    </td>
                    <td style={{ ...S.td, color:"#64748b" }}>#{lRank}</td>
                    <td style={S.td}>
                      {delta === 0
                        ? <span style={{ color:"#94a3b8" }}>—</span>
                        : <span style={S.pill(delta >= 4 ? "#ef4444" : delta >= 2 ? "#f59e0b" : "#64748b")}>
                            Δ{delta}
                          </span>}
                    </td>
                    <td style={S.td}>
                      <button style={{ ...S.btn("#3b82f6"), marginRight:3, padding:"3px 7px" }}
                              onClick={() => ask(q, "eig")}>EIG</button>
                      <button style={{ ...S.btn("#ef4444"), padding:"3px 7px" }}
                              onClick={() => ask(q, "ellr")}>ELLR</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════
           SIMULATION RESULTS
         ══════════════════════════════════════════════════════════ */}
      {simRes && (
        <div style={S.card}>
          <div style={S.h2}>
            Simulation — {N_SIM} games per (hypothesis × strategy)
          </div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:8 }}>
            Each game: always ask the highest-scoring question per criterion.
            Converge when max belief &gt; {(threshold*100).toFixed(0)}%. Cap at {MAX_Q} questions.
            Accuracy = fraction of games where the highest-weight hypothesis equals the true one.
          </div>
          <table style={{ width:"100%", borderCollapse:"collapse" }}>
            <thead>
              <tr>
                {["Hypothesis (prior)","EIG avg Q","EIG acc","E[logLR] avg Q","E[logLR] acc","Fewer Q","Highest acc"].map(h => (
                  <th key={h} style={S.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {HYPS.map((h, i) => {
                const re = simRes.eig.byHyp[i],  rl = simRes.ellr.byHyp[i];
                const aqE = re.nq/re.n,           aqL = rl.nq/rl.n;
                const acE = re.ok/re.n,           acL = rl.ok/rl.n;
                const WIN_MARGIN = 0.08;
                const win = aqE < aqL - WIN_MARGIN ? "EIG"
                          : aqL < aqE - WIN_MARGIN ? "E[logLR]"
                          : "Tie";
                const ACC_MARGIN = 0.005;
                const winAcc = acE > acL + ACC_MARGIN ? "EIG"
                             : acL > acE + ACC_MARGIN ? "E[logLR]"
                             : "Tie";
                return (
                  <tr key={h.id}>
                    <td style={S.td}>
                      <span style={S.pill(h.color)}>{h.label}</span>
                      {" "}{h.desc}{" "}
                      <span style={{ color:"#94a3b8" }}>(prior {h.prior})</span>
                    </td>
                    <td style={{ ...S.td, color:"#3b82f6", fontWeight: win==="EIG" ? 700 : "normal" }}>
                      {aqE.toFixed(2)}
                    </td>
                    <td style={{ ...S.td, color:"#3b82f6" }}>{(acE*100).toFixed(1)}%</td>
                    <td style={{ ...S.td, color:"#ef4444", fontWeight: win==="E[logLR]" ? 700 : "normal" }}>
                      {aqL.toFixed(2)}
                    </td>
                    <td style={{ ...S.td, color:"#ef4444" }}>{(acL*100).toFixed(1)}%</td>
                    <td style={S.td}>
                      <span style={S.pill(win==="EIG"?"#3b82f6":win==="E[logLR]"?"#ef4444":"#22c55e")}>{win}</span>
                    </td>
                    <td style={S.td}>
                      <span style={S.pill(winAcc==="EIG"?"#3b82f6":winAcc==="E[logLR]"?"#ef4444":"#22c55e")}>{winAcc}</span>
                    </td>
                  </tr>
                );
              })}
              {(() => {
                // Compute overall as either equal-weighted or prior-weighted
                let aqE, aqL, acE, acL;
                if (priorWeighted) {
                  const totalPrior = HYPS.reduce((s, h) => s + h.prior, 0);
                  aqE = HYPS.reduce((s, h, i) => s + h.prior * simRes.eig.byHyp[i].nq  / simRes.eig.byHyp[i].n,  0) / totalPrior;
                  aqL = HYPS.reduce((s, h, i) => s + h.prior * simRes.ellr.byHyp[i].nq / simRes.ellr.byHyp[i].n, 0) / totalPrior;
                  acE = HYPS.reduce((s, h, i) => s + h.prior * simRes.eig.byHyp[i].ok  / simRes.eig.byHyp[i].n,  0) / totalPrior;
                  acL = HYPS.reduce((s, h, i) => s + h.prior * simRes.ellr.byHyp[i].ok / simRes.ellr.byHyp[i].n, 0) / totalPrior;
                } else {
                  const re = simRes.eig.total, rl = simRes.ellr.total;
                  aqE = re.nq/re.n; aqL = rl.nq/rl.n;
                  acE = re.ok/re.n; acL = rl.ok/rl.n;
                }
                const WIN_MARGIN = 0.08;
                const win = aqE < aqL - WIN_MARGIN ? "EIG"
                           : aqL < aqE - WIN_MARGIN ? "E[logLR]"
                           : "Tie";
                const ACC_MARGIN = 0.005;
                const winAcc = acE > acL + ACC_MARGIN ? "EIG"
                             : acL > acE + ACC_MARGIN ? "E[logLR]"
                             : "Tie";
                return (
                  <tr style={{ background:"#f1f5f9", fontWeight:700 }}>
                    <td style={S.td}>
                      <span style={{ marginRight:8 }}>
                        OVERALL
                      </span>
                      <label style={{ fontWeight:400, fontSize:11, color:"#475569", cursor:"pointer", userSelect:"none" }}>
                        <input
                          type="checkbox"
                          checked={priorWeighted}
                          onChange={e => setPriorWeighted(e.target.checked)}
                          style={{ marginRight:4, accentColor:"#3b82f6" }}
                        />
                        prior-weighted
                        <span style={{ color:"#94a3b8", marginLeft:4 }}>
                          {priorWeighted
                            ? "(H1×0.45 + H2×0.45 + H3×0.05 + H4×0.05)"
                            : "(H1×0.25 + H2×0.25 + H3×0.25 + H4×0.25)"}
                        </span>
                      </label>
                    </td>
                    <td style={{ ...S.td, color:"#3b82f6" }}>{aqE.toFixed(2)}</td>
                    <td style={{ ...S.td, color:"#3b82f6" }}>{(acE*100).toFixed(1)}%</td>
                    <td style={{ ...S.td, color:"#ef4444" }}>{aqL.toFixed(2)}</td>
                    <td style={{ ...S.td, color:"#ef4444" }}>{(acL*100).toFixed(1)}%</td>
                    <td style={S.td}>
                      <span style={S.pill(win==="EIG"?"#3b82f6":win==="E[logLR]"?"#ef4444":"#22c55e")}>{win}</span>
                    </td>
                    <td style={S.td}>
                      <span style={S.pill(winAcc==="EIG"?"#3b82f6":winAcc==="E[logLR]"?"#ef4444":"#22c55e")}>{winAcc}</span>
                    </td>
                  </tr>
                );
              })()}
            </tbody>
          </table>
          <div style={{ marginTop:10, fontSize:11, color:"#64748b", lineHeight:1.7 }}>
            <b>What to look for:</b> When the true hypothesis is H1 or H2 (prior 0.45 each),
            E[log LR] should converge faster because it asks cell questions that sharply discriminate
            the dominant hypothesis. When H3 or H4 is true (prior 0.05), EIG may edge ahead because
            its row question provides information about the column-2 ship too.
            The overall winner depends on how the prior weights interact with the question structure.
          </div>
        </div>
      )}

    </div>
  );
}
