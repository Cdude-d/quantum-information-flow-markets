"""HTML morning report + console summary."""

from datetime import datetime
from pathlib import Path

CSS = """
:root { --bg:#f4f6f9; --card:#fff; --ink:#1a2233; --muted:#5b6678; --line:#e3e8ef;
        --accent:#2456c9; --green:#147a3d; --red:#b3261e; --amber:#8a5a00; }
* { box-sizing: border-box; }
body { margin:0; padding:28px 18px; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size:24px; margin:0 0 2px; } h2 { font-size:18px; margin:28px 0 10px; }
.sub { color:var(--muted); margin-bottom:18px; }
.box { background:var(--card); border:1px solid var(--line); border-radius:12px;
       padding:16px 18px; margin-bottom:14px; }
.context { border-left:4px solid var(--accent); }
.note { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:18px 20px; margin-bottom:16px; }
.card h3 { margin:0; font-size:17px; }
.head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
         font-weight:600; background:#eef2fb; color:var(--accent); }
.badge.sell { background:#e9f6ee; color:var(--green); }
.badge.buy  { background:#eef2fb; color:var(--accent); }
.score { font-size:13px; color:var(--muted); }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(118px,1fr));
           gap:10px; margin:14px 0; }
.m { background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
.m .k { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.m .v { font-size:16px; font-weight:650; margin-top:2px; }
.v.pos { color:var(--green); } .v.neg { color:var(--red); }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:8px 0; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }
.reason p { margin:8px 0; } .reason b { color:var(--ink); }
.warn { color:var(--amber); font-weight:550; }
.disclaimer { font-size:12.5px; color:var(--muted); border-top:1px solid var(--line);
              margin-top:26px; padding-top:14px; }
.empty { color:var(--muted); font-style:italic; }
.win { color:var(--green); font-weight:600; } .loss { color:var(--red); font-weight:600; }
"""

MODE_TITLES = {"eod": "End of Day (0DTE)", "eow": "End of Week"}


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def _card(rank: int, c: dict, reasoning: dict) -> str:
    sell = c["net"] >= 0
    badge = f'<span class="badge {"sell" if sell else "buy"}">{"CREDIT" if sell else "DEBIT"}</span>'
    strikes = "/".join(f"{l['k']:g}" for l in c["legs"])
    ev_cls = "pos" if c["ev_model"] >= 0 else "neg"
    legs_rows = "".join(
        f"<tr><td>{l['side'].upper()}</td><td>{l['type'].upper()}</td><td>{l['k']:g}</td>"
        f"<td>{l['bid']:.2f} × {l['ask']:.2f}</td><td>{l['mid']:.2f}</td>"
        f"<td>{l['delta']:.2f}</td><td>{l['iv']*100:.0f}%</td><td>{l['liq']:,}</td></tr>"
        for l in c["legs"])
    warns = "".join(f'<p class="warn">⚠ {w}</p>' for w in c["warnings"])
    em = (f"±{c['expected_move']:.2f}" if c["expected_move"] else "n/a")
    sizing = ""
    if c["sizing"]:
        sizing = (f'<div class="m"><div class="k">Suggested size</div>'
                  f'<div class="v">{c["sizing"]["contracts"]}x</div></div>')
    return f"""
<div class="card">
  <div class="head">
    <h3>#{rank} &nbsp;{c['symbol']} — {c['structure']} {strikes} {badge}</h3>
    <span class="score">exp {c['expiry']:%a %b %d} · score {c['score']}</span>
  </div>
  <div class="metrics">
    <div class="m"><div class="k">POP (model)</div><div class="v">{c['pop_model']*100:.0f}%</div></div>
    <div class="m"><div class="k">POP (market)</div><div class="v">{c['pop_market']*100:.0f}%</div></div>
    <div class="m"><div class="k">{'Credit' if sell else 'Debit'} / contract</div>
        <div class="v">{_money(abs(c['net'])*100)}</div></div>
    <div class="m"><div class="k">Max gain</div><div class="v pos">{_money(c['max_gain'])}</div></div>
    <div class="m"><div class="k">Max loss</div><div class="v neg">{_money(c['max_loss'])}</div></div>
    <div class="m"><div class="k">Model EV</div><div class="v {ev_cls}">{_money(c['ev_model'])}</div></div>
    <div class="m"><div class="k">Breakeven</div><div class="v">{", ".join(f"{b:g}" for b in c['breakevens'])}</div></div>
    <div class="m"><div class="k">Expected move</div><div class="v">{em}</div></div>
    {sizing}
  </div>
  <table>
    <tr><th>Side</th><th>Type</th><th>Strike</th><th>Bid × Ask</th><th>Mid</th>
        <th>|Δ|</th><th>IV</th><th>OI/Vol</th></tr>
    {legs_rows}
  </table>
  <div class="reason">
    <p><b>Setup.</b> {reasoning['setup']}</p>
    <p><b>Volatility.</b> {reasoning['volatility']}</p>
    <p><b>Trade.</b> {reasoning['trade']}</p>
    <p><b>Risk.</b> {reasoning['risk']}</p>
    {warns}
  </div>
</div>"""


def _scoreboard_html(stats: dict) -> str:
    if not stats["n_scored"]:
        return ('<div class="box"><span class="empty">No completed trades yet — the track '
                'record builds automatically as past recommendations reach expiry.</span></div>')
    wr = stats["win_rate"] * 100
    rows = "".join(
        f"<tr><td>{r['date']}</td><td>{r['symbol']}</td><td>{r['structure']}</td>"
        f"<td>{r['mode'].upper()}</td><td>{r['pop_model']*100:.0f}%</td>"
        f"<td class=\"{'win' if r['win'] else 'loss'}\">{_money(r['pl'])}</td></tr>"
        for r in reversed(stats["recent"]))
    return f"""
<div class="box">
  <p><b>{stats['n_scored']}</b> recommendations scored · win rate <b>{wr:.0f}%</b> ·
     total P/L <b>{_money(stats['total_pl'])}</b> per 1-lot ·
     avg return on risk <b>{stats['avg_r']*100:+.0f}%</b>
     <span class="note">(held-to-expiry assumption, no profit-taking)</span></p>
  <table>
    <tr><th>Picked</th><th>Symbol</th><th>Structure</th><th>Mode</th><th>POP est.</th><th>P/L</th></tr>
    {rows}
  </table>
</div>"""


def render_html(now_et: datetime, session_date, context_text: str,
                picks: dict[str, list], reasonings: dict[int, dict],
                stats: dict, notes: list[str]) -> str:
    sections = []
    for mode in ("eod", "eow"):
        if mode not in picks:
            continue
        items = picks[mode]
        body = ('<div class="box"><span class="empty">No trades passed the filters for this '
                'horizon today — that is a valid answer; forcing a trade is how edges die.'
                '</span></div>')
        if items:
            body = "".join(_card(i + 1, c, reasonings[id(c)]) for i, c in enumerate(items))
        sections.append(f"<h2>{MODE_TITLES[mode]}</h2>{body}")

    notes_html = ""
    if notes:
        notes_html = ('<div class="box note">' +
                      "<br>".join(f"• {n}" for n in notes) + "</div>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Options Morning Scout — {session_date:%b %d, %Y}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Options Morning Scout</h1>
<div class="sub">Session: {session_date:%A, %B %d, %Y} · generated {now_et:%a %b %d, %I:%M %p} ET</div>
<div class="box context"><b>Market context.</b> {context_text}</div>
{''.join(sections)}
<h2>Track record</h2>
{_scoreboard_html(stats)}
{notes_html}
<div class="disclaimer">Educational decision support, not financial advice. Probabilities are
lognormal-model estimates from delayed quotes (~15 min) and can be wrong, especially through
events; expected values are model opinions, not promises. Options involve substantial risk of
loss — 0DTE structures can lose their full max loss in hours. Verify prices in your broker
before trading, use limit orders at or near the listed mid, and size so a full loss is
acceptable. Past scanner performance does not guarantee future results.</div>
</div></body></html>"""


def console_summary(session_date, context_text: str, picks: dict[str, list],
                    stats: dict, report_path: Path) -> str:
    lines = [f"=== Options Morning Scout — session {session_date:%a %b %d, %Y} ===",
             context_text, ""]
    for mode in ("eod", "eow"):
        if mode not in picks:
            continue
        lines.append(f"--- {MODE_TITLES[mode]} ---")
        if not picks[mode]:
            lines.append("  (no trades passed filters)")
        for i, c in enumerate(picks[mode], 1):
            strikes = "/".join(f"{l['k']:g}" for l in c["legs"])
            net = f"{'cr' if c['net'] >= 0 else 'db'} {abs(c['net']):.2f}"
            lines.append(
                f"  {i}. {c['symbol']} {c['structure']} {strikes} exp {c['expiry']:%m-%d} "
                f"({net}) | POP {c['pop_model']*100:.0f}% | EV {c['ev_model']:+.0f} "
                f"| score {c['score']}")
        lines.append("")
    if stats["n_scored"]:
        lines.append(f"Track record: {stats['n_scored']} scored, "
                     f"win rate {stats['win_rate']*100:.0f}%, "
                     f"total P/L {stats['total_pl']:+,.0f} per 1-lot")
    lines.append(f"Report: {report_path}")
    return "\n".join(lines)
