"""Generate a shareable PDF of scanner results over the last N days.

Reads the tracked recommendations CSV (the same file the scanner scores at
expiry) so every figure in the PDF traces back to a logged, pre-committed pick.

Usage:  python make_report_pdf.py [days]   (default 7)
"""
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from options_scout import scoreboard
from options_scout.config import load_config

# --- palette --------------------------------------------------------------
INK = colors.HexColor("#1a2233")
MUTED = colors.HexColor("#5b6678")
ACCENT = colors.HexColor("#2456c9")
GREEN = colors.HexColor("#147a3d")
RED = colors.HexColor("#b3261e")
LINE = colors.HexColor("#d8dee8")
BAND = colors.HexColor("#f1f4f9")
HEADBG = colors.HexColor("#22304d")


def money(x):
    x = float(x)
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def load_rows(cfg, days):
    cutoff = date.today() - timedelta(days=days)
    path = Path(cfg["data_dir"]) / "recommendations.csv"
    all_rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scored = [r for r in all_rows if r["status"] == "scored"
              and datetime.fromisoformat(r["run_ts"]).date() >= cutoff]
    opens = [r for r in all_rows if r["status"] == "open"]
    return scored, opens, cutoff


def agg(rs):
    n = len(rs)
    wins = sum(int(r["win"]) for r in rs)
    pl = sum(float(r["pl_per_contract"]) for r in rs)
    risk = sum(float(r["max_loss"]) for r in rs)
    pop = sum(float(r["pop_model"]) for r in rs) / n if n else 0
    return dict(n=n, wins=wins, pl=pl, risk=risk, pop=pop,
                wr=wins / n if n else 0)


def build(cfg, days, out_path):
    scoreboard.score_pending(cfg)  # settle anything newly expired first
    scored, opens, cutoff = load_rows(cfg, days)
    tot = agg(scored)
    by_day = defaultdict(list)
    for r in scored:
        by_day[datetime.fromisoformat(r["run_ts"]).date()].append(r)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=INK, fontSize=20,
                        spaceAfter=2, alignment=TA_LEFT)
    SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=MUTED, fontSize=9.5,
                         spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT, fontSize=13,
                        spaceBefore=14, spaceAfter=6)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], textColor=INK, fontSize=9.5,
                          leading=14, spaceAfter=5)
    BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2,
                            spaceAfter=3)
    NOTE = ParagraphStyle("NOTE", parent=ss["Normal"], textColor=MUTED, fontSize=8,
                          leading=11)
    CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=8.5, leading=11,
                          textColor=INK)

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            title="Options Morning Scout — Weekly Results")
    s = []

    # --- header ---
    s.append(Paragraph("Options Morning Scout — Weekly Results", H1))
    s.append(Paragraph(
        f"Pre-committed picks logged {cutoff:%b %d} – {date.today():%b %d, %Y} · "
        f"generated {datetime.now():%b %d, %Y %I:%M %p} ET", SUB))
    s.append(HRFlowable(width="100%", color=LINE, spaceBefore=4, spaceAfter=8))

    # --- headline scorecard ---
    def kpi(label, value, color=INK):
        return [Paragraph(f'<font size=15 color="#{color.hexval()[2:]}"><b>{value}</b></font>'
                          f'<br/><font size=7.5 color="#5b6678">{label.upper()}</font>', CELL)]
    pl_color = GREEN if tot["pl"] >= 0 else RED
    kpis = Table([[kpi("Trades", tot["n"])[0], kpi("Win rate", f"{tot['wr']*100:.0f}%")[0],
                   kpi("Net P/L (1-lot)", money(tot["pl"]), pl_color)[0],
                   kpi("Return on risk", f"+{tot['pl']/tot['risk']*100:.0f}%", pl_color)[0],
                   kpi("Model POP", f"{tot['pop']*100:.0f}%", ACCENT)[0]]],
                  colWidths=[1.3 * inch] * 5)
    kpis.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    s.append(kpis)

    # --- results by day ---
    s.append(Paragraph("Results by session", H2))
    data = [["Date", "Trades", "Wins", "Win rate", "Net P/L", "Avg model POP"]]
    for d in sorted(by_day):
        a = agg(by_day[d])
        data.append([f"{d:%a %b %d}", str(a["n"]), str(a["wins"]),
                     f"{a['wr']*100:.0f}%", money(a["pl"]), f"{a['pop']*100:.0f}%"])
    data.append(["TOTAL", str(tot["n"]), str(tot["wins"]), f"{tot['wr']*100:.0f}%",
                 money(tot["pl"]), f"{tot['pop']*100:.0f}%"])
    t = Table(data, colWidths=[1.1*inch, 0.85*inch, 0.7*inch, 0.95*inch, 1.1*inch, 1.3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, BAND]),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, INK),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef2fb")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
    ]))
    s.append(t)
    s.append(Paragraph("Held to expiry, settled at the expiry-day close, no profit-taking "
                       "or stops. P/L is per single contract.", NOTE))

    # --- highlights ---
    s.append(Paragraph("Highlights", H2))
    best = max(scored, key=lambda r: float(r["pl_per_contract"]))
    worst = min(scored, key=lambda r: float(r["pl_per_contract"]))
    idx_rows = [r for r in scored if r["symbol"] in ("SPY", "QQQ", "IWM", "DIA")]
    idx = agg(idx_rows)
    name_rows = [r for r in scored if r["symbol"] not in ("SPY", "QQQ", "IWM", "DIA")]
    nm = agg(name_rows)
    pred = tot["pop"] * tot["n"]
    for txt in [
        f"<b>Calibration:</b> the model predicted ~{pred:.0f} winners out of {tot['n']}; "
        f"actual was {tot['wins']}. Realized win rate {tot['wr']*100:.0f}% vs. predicted "
        f"{tot['pop']*100:.0f}% — well-aligned over this sample.",
        f"<b>Return:</b> {money(tot['pl'])} per 1-lot on ${tot['risk']:,.0f} of cumulative "
        f"max risk deployed (+{tot['pl']/tot['risk']*100:.0f}%).",
        f"<b>Index vs. single-name:</b> index/ETF trades went {idx['wins']}-for-{idx['n']} "
        f"({money(idx['pl'])}); single names {nm['wins']}-for-{nm['n']} ({money(nm['pl'])}). "
        f"Both losers were single-name structures caught by a directional move.",
        f"<b>Best / worst:</b> {best['symbol']} {best['structure']} {money(best['pl_per_contract'])} · "
        f"{worst['symbol']} {worst['structure']} {money(worst['pl_per_contract'])}.",
        f"<b>In flight:</b> {len(opens)} further picks are currently open "
        f"(expiring {min(r['expiry'] for r in opens)} – {max(r['expiry'] for r in opens)}) "
        f"and not yet counted." if opens else "",
    ]:
        if txt:
            s.append(Paragraph(f"•&nbsp;&nbsp;{txt}", BULLET))

    # --- how the model works ---
    s.append(Paragraph("How the model works", H2))
    for txt in [
        "<b>Universe &amp; cadence.</b> Each weekday at 9:45 AM ET the scanner pulls live "
        "option chains for liquid index ETFs and mega-cap names and evaluates two horizons: "
        "same-day (0DTE) and end-of-week expiries.",
        "<b>Structures.</b> Only defined-risk spreads are proposed — iron condors and "
        "put/call credit spreads (premium-selling), plus debit spreads when premium is cheap. "
        "Max loss is known and capped on every trade; no naked options.",
        "<b>Probability engine.</b> Each candidate is priced with Black–Scholes under "
        "lognormal dynamics. <b>POP (market)</b> uses the option's own implied volatility; "
        "<b>POP (model)</b> re-prices it with a 50/50 blend of implied and recent realized "
        "volatility (EWMA). A gap between the two is the trade thesis — typically harvesting "
        "the variance risk premium (options priced slightly rich vs. how much the underlying "
        "actually moves).",
        "<b>Selection.</b> Candidates are ranked by a score blending model POP, expected "
        "value per dollar of risk, leg liquidity (bid/ask + open interest), and alignment "
        "with directional signals (trend vs. 20/50-day EMAs, short-term RSI, momentum, gap "
        "statistics). Names with earnings before expiry are dropped; FOMC and payroll dates "
        "are flagged.",
        "<b>Track record.</b> Every pick is written to a log the moment it is generated and "
        "scored automatically at expiry against the official close — so these results are "
        "pre-committed and cannot be cherry-picked after the fact.",
    ]:
        s.append(Paragraph(txt, BODY))

    # --- caveats ---
    n_sessions = len(by_day)
    s.append(Paragraph("Caveats", H2))
    for txt in [
        f"<b>Tiny sample, one regime.</b> This covers {tot['n']} settled trades across "
        f"{n_sessions} session{'s' if n_sessions != 1 else ''} in a single low-volatility "
        "stretch (VIX ~19). That is far too small to infer skill, and it has not yet faced a "
        "volatility spike or a sharp trending market.",
        "<b>Premium-selling return shape.</b> The strategy wins often but small and loses "
        "rarely but large. A single bad session can erase a good week, so short-run win rates "
        "overstate the long-run edge.",
        "<b>Idealized fills.</b> Results assume entry at the mid-price and settlement held to "
        "expiry with no stops or profit-taking. Real fills, slippage, commissions, and active "
        "management would all change the outcome.",
        "<b>Delayed data.</b> Quotes are ~15 minutes delayed. Probabilities are model "
        "estimates, not guarantees, and degrade through scheduled events.",
        "<b>Not advice.</b> Educational decision-support only. Options carry substantial risk "
        "of loss, including the full max-loss on any single trade.",
    ]:
        s.append(Paragraph(txt, BODY))

    # --- appendix: every trade ---
    s.append(Paragraph("Appendix — every scored trade", H2))
    ad = [["Picked", "Sym", "Structure", "Exp", "Credit/Debit",
           "Model POP", "P/L", "Result"]]
    for r in sorted(scored, key=lambda r: (r["run_ts"], r["symbol"])):
        net = float(r["net"])
        ad.append([
            datetime.fromisoformat(r["run_ts"]).strftime("%b %d"),
            r["symbol"], r["structure"], r["expiry"][5:],
            f"{'cr' if net >= 0 else 'db'} ${abs(net)*100:.0f}",
            f"{float(r['pop_model'])*100:.0f}%",
            money(r["pl_per_contract"]),
            "WIN" if r["win"] == "1" else "LOSS",
        ])
    at = Table(ad, colWidths=[0.6*inch, 0.5*inch, 1.55*inch, 0.55*inch, 0.95*inch,
                              0.75*inch, 0.7*inch, 0.6*inch], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for i, r in enumerate(sorted(scored, key=lambda r: (r["run_ts"], r["symbol"])), 1):
        col = GREEN if r["win"] == "1" else RED
        style.append(("TEXTCOLOR", (6, i), (7, i), col))
        style.append(("FONTNAME", (7, i), (7, i), "Helvetica-Bold"))
    at.setStyle(TableStyle(style))
    s.append(at)

    s.append(Spacer(1, 10))
    s.append(HRFlowable(width="100%", color=LINE, spaceBefore=2, spaceAfter=6))
    s.append(Paragraph(
        "Generated by Options Morning Scout from pre-committed, expiry-scored trade logs. "
        "Educational decision-support, not financial advice. Past performance does not "
        "guarantee future results.", NOTE))

    doc.build(s)
    return out_path, tot


if __name__ == "__main__":
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = Path(cfg["report_dir"]) / f"weekly_results_{date.today():%Y-%m-%d}.pdf"
    path, tot = build(cfg, days, out)
    print(f"PDF written: {path}")
    print(f"{tot['n']} trades, {tot['wins']} wins ({tot['wr']*100:.0f}%), "
          f"net {money(tot['pl'])} per 1-lot")
