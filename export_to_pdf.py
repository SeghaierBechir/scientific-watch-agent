"""Export Scientific Watch Agent pipeline results to a professional PDF report.

The generated PDF contains 7 sections ordered from most actionable to most
technical — so a researcher opening the report sees insights first:

    Cover page  (topic, run metadata, token usage, quality score)
    1. Trends, Gaps & Future Perspectives
       — Research trends table (emerging / established / declining)
       — Emerging trends: supporting article titles + 2-3 sentence digest
       — Research gaps with importance badge + suggested directions
       — Future perspectives bullet list
    2. Global Synthesis  (overview, main approaches, datasets, key findings)
    3. Top Articles  (ranked table + sub-score breakdown)
    4. Article Summaries  (card per article: problem/method/dataset/results)
    5. Query Expansion  (expanded queries list from ReAct loop)
    6. Reflexion Loop  (per-iteration critic quality, issues, suggestions)
    7. Agent Execution Logs  (duration, tokens, API calls per agent)

Requires:  pip install reportlab

Usage (standalone):
    python export_to_pdf.py "fake news detection"
    python export_to_pdf.py "fake news detection" 30 5
    python export_to_pdf.py "fake news detection" 30 5 --output my_report.pdf

Usage (as library — import into demo scripts):
    from export_to_pdf import export_to_pdf
    pdf_path = export_to_pdf(final_state)           # auto-named
    pdf_path = export_to_pdf(final_state, "out.pdf")
"""

from __future__ import annotations

import argparse
import logging
import smtplib
import sys
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether,
    )
except ImportError:
    print("ERROR: reportlab is not installed.  Run:  pip install reportlab")
    sys.exit(1)

from src.agents.graph import run_pipeline
from src.config import (
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    EMAIL_SENDER,
    MAX_REFLEXION_ITERATIONS,
    REFLEXION_MIN_QUALITY,
    SMTP_HOST,
    SMTP_PORT,
)
from src.schemas import AgentLog, CriticFeedback

logger = logging.getLogger(__name__)

# ── Page geometry ─────────────────────────────────────────────────────────────
_PW, _PH = A4
_LR = 2.0 * cm          # left/right margin
_TB = 2.0 * cm          # top/bottom margin
CW  = _PW - 2 * _LR    # usable content width  (~170 mm on A4)

# ── Color palette ─────────────────────────────────────────────────────────────
CN   = HexColor('#1a3a5c')   # navy  — section headers
CB   = HexColor('#2d6a9f')   # blue  — table headers, sub-headers
CBL  = HexColor('#dce9f5')   # light blue — key-column backgrounds
CGL  = HexColor('#f4f5f6')   # light gray — alternating table rows
CGM  = HexColor('#cccccc')   # medium gray — grid lines / HR
CGD  = HexColor('#555555')   # dark gray — secondary text
CGR  = HexColor('#27ae60')   # green — "good/excellent" quality, OK status
COR  = HexColor('#e67e22')   # orange — "acceptable" quality, medium importance
CRD  = HexColor('#c0392b')   # red — "poor" quality, high importance, errors
CTE  = HexColor('#16a085')   # teal — "excellent" quality
CGO  = HexColor('#c8960c')   # gold — scores

# Maps for semantic coloring
QCOL = {'poor': CRD, 'acceptable': COR, 'good': CGR, 'excellent': CTE}
ICOL = {'low': CGD, 'medium': COR, 'high': CRD}
MCOL = {'emerging': CGR, 'established': CB, 'declining': COR}


# ── XML/HTML escaping ─────────────────────────────────────────────────────────

def _esc(text: str | None) -> str:
    """Escape characters that would break ReportLab's Paragraph XML parser."""
    if not text:
        return ''
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )


def _hex(color) -> str:
    """Convert a ReportLab Color to a CSS hex string for inline markup."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(color.red * 255), int(color.green * 255), int(color.blue * 255)
    )


def _colored(text: str, color) -> str:
    """Wrap text in a colored bold <font> tag (for Paragraph markup)."""
    return f'<font color="{_hex(color)}"><b>{_esc(text)}</b></font>'


# ── Style registry ────────────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    """Build all named ParagraphStyles, deriving from base 'Normal'."""
    N = getSampleStyleSheet()['Normal']

    def ps(name: str, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=N, **kw)

    return {
        # ── Cover page ─────────────────────────────────────────────────────
        'cov_h1':   ps('cov_h1',  fontSize=28, fontName='Helvetica-Bold',
                        textColor=CN, alignment=TA_CENTER, leading=34, spaceAfter=6),
        'cov_sub':  ps('cov_sub', fontSize=16, fontName='Helvetica-BoldOblique',
                        textColor=CB, alignment=TA_CENTER, spaceAfter=6),
        'cov_meta': ps('cov_meta', fontSize=9, textColor=CGD,
                        alignment=TA_CENTER, spaceAfter=3),

        # ── Section / sub-section headers ──────────────────────────────────
        'sec_h1':   ps('sec_h1', fontSize=13, fontName='Helvetica-Bold',
                        textColor=white, spaceBefore=0, spaceAfter=0),
        'sec_h2':   ps('sec_h2', fontSize=10, fontName='Helvetica-Bold',
                        textColor=CN, spaceBefore=8, spaceAfter=4),
        'sec_h3':   ps('sec_h3', fontSize=9,  fontName='Helvetica-Bold',
                        textColor=CB, spaceBefore=4, spaceAfter=2),

        # ── Body text ──────────────────────────────────────────────────────
        'body':     ps('body', fontSize=9, textColor=black, leading=13,
                        alignment=TA_JUSTIFY, spaceAfter=4),
        'body_sm':  ps('body_sm', fontSize=8, textColor=CGD, leading=11),

        # ── Bullet list items ──────────────────────────────────────────────
        'li':       ps('li', fontSize=9, textColor=black, leading=13,
                        leftIndent=14, firstLineIndent=-10, spaceAfter=2),

        # ── Table cells ────────────────────────────────────────────────────
        'tc':       ps('tc',    fontSize=8, textColor=black,  leading=10),
        'tc_b':     ps('tc_b',  fontSize=8, fontName='Helvetica-Bold',
                        textColor=CN, leading=10),
        'tc_c':     ps('tc_c',  fontSize=8, textColor=black,
                        leading=10, alignment=TA_CENTER),
        'tc_bs':    ps('tc_bs', fontSize=7, textColor=CGD, leading=9),

        # ── Score / metric ─────────────────────────────────────────────────
        'score':    ps('score', fontSize=9, fontName='Helvetica-Bold',
                        textColor=CGO, alignment=TA_CENTER),
    }


_STYLES: dict[str, ParagraphStyle] = {}


def _st() -> dict[str, ParagraphStyle]:
    """Lazy-load style registry (avoids building at import time)."""
    global _STYLES
    if not _STYLES:
        _STYLES = _build_styles()
    return _STYLES


# ── Reusable flowable helpers ─────────────────────────────────────────────────

def _section_header(title: str, num: str | None = None) -> list:
    """Full-width navy bar used as section header."""
    label = f"  {num}. {title}" if num else f"  {title}"
    inner = Paragraph(label, _st()['sec_h1'])
    bar = Table([[inner]], colWidths=[CW])
    bar.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), CN),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    return [Spacer(1, 0.35 * cm), bar, Spacer(1, 0.25 * cm)]


def _li(text: str) -> Paragraph:
    """Single bullet-list item."""
    return Paragraph(f'&#x2022;  {text}', _st()['li'])


def _sub_li(text: str) -> Paragraph:
    """Indented sub-bullet item (for suggested directions etc.)."""
    return Paragraph(f'&nbsp;&nbsp;&nbsp;&nbsp;&#x25B8;  {text}', _st()['li'])


def _hr() -> HRFlowable:
    return HRFlowable(width=CW, color=CGM, thickness=0.3,
                      spaceBefore=3, spaceAfter=3)


def _kv_table(pairs: list[tuple[str, str]], col_ratio: float = 0.40) -> Table:
    """Two-column key-value table used in the cover page."""
    st = _st()
    rows = [
        [Paragraph(k, st['tc_b']), Paragraph(v, st['tc'])]
        for k, v in pairs
    ]
    t = Table(rows, colWidths=[CW * col_ratio, CW * (1 - col_ratio)])
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
    ]))
    return t


# ── Table style factory ───────────────────────────────────────────────────────

def _base_table_style(n_rows: int, has_total_row: bool = False) -> TableStyle:
    """Standard style: blue header, alternating gray rows, light grid."""
    cmds = [
        ('BACKGROUND',    (0, 0), (-1, 0),  CB),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  8),
        ('GRID',          (0, 0), (-1, -1), 0.3, CGM),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
    ]
    # Alternating rows (skip header row 0, and total row if any)
    last = n_rows - 2 if has_total_row else n_rows - 1
    for i in range(2, last + 1, 2):
        cmds.append(('BACKGROUND', (0, i), (-1, i), CGL))
    if has_total_row:
        cmds.append(('BACKGROUND', (0, n_rows - 1), (-1, n_rows - 1), CBL))
    return TableStyle(cmds)


# ── Section builders ──────────────────────────────────────────────────────────

def _build_cover(final: dict) -> list:
    """Cover page: report title, topic, and run metadata."""
    st   = _st()
    els  = [Spacer(1, 2.2 * cm)]
    topic = final.get('topic', 'Unknown Topic')

    els.append(Paragraph('Scientific Watch Report agents ReAct+Reflexion', st['cov_h1']))
    els.append(Spacer(1, 0.3 * cm))
    els.append(Paragraph(f'&ldquo;{_esc(topic)}&rdquo;', st['cov_sub']))
    els.append(Spacer(1, 1.0 * cm))
    els.append(HRFlowable(width=CW * 0.5, color=CB, thickness=1.5,
                           spaceAfter=6, hAlign='CENTER'))
    els.append(Spacer(1, 0.8 * cm))

    cfg        = final.get('config', {})
    n_iter     = final.get('synthesis_iteration', 1)
    n_feedbacks = len(final.get('critic_feedbacks', []))
    total_tok  = sum(lg.tokens_used for lg in final.get('logs', []))
    last_fb    = (final.get('critic_feedbacks') or [None])[-1]
    final_qual = last_fb.overall_quality.upper() if last_fb else 'N/A'

    meta: list[tuple[str, str]] = [
        ('Generated on',         datetime.now().strftime('%d %B %Y at %H:%M')),
        ('Topic',                topic),
        ('Articles fetched',     str(cfg.get('n_raw', '?'))),
        ('Articles selected',    str(len(final.get('top_articles', [])))),
        ('Summaries produced',   str(len(final.get('summaries', [])))),
        ('Reflexion iterations', f"{n_iter} / {MAX_REFLEXION_ITERATIONS} max"),
        ('Critic evaluations',   str(n_feedbacks)),
        ('Final synthesis quality', final_qual),
        ('Min. quality threshold',  REFLEXION_MIN_QUALITY.upper()),
        ('Total tokens used',    f"{total_tok:,}"),
    ]
    els.append(_kv_table(meta, col_ratio=0.42))
    els.append(PageBreak())
    return els


def _build_queries(final: dict, sec_num: str) -> list:
    """Section 1 — Query Expansion."""
    st      = _st()
    queries = final.get('expanded_queries') or [final.get('topic', '')]
    els     = _section_header(f"Query Expansion  ({len(queries)} queries)", sec_num)

    els.append(Paragraph(
        'The <b>QueryExpander</b> agent produces diverse search queries from the original '
        'topic, maximising coverage of different angles, synonyms, and sub-tasks. '
        'OpenAlex is searched independently for each query; results are then deduplicated.',
        st['body'],
    ))
    els.append(Spacer(1, 0.2 * cm))

    for i, q in enumerate(queries, 1):
        tag = (
            ' <font size="7" color="#888888">[original]</font>' if i == 1
            else f' <font size="7" color="#888888">[variant {i - 1}]</font>'
        )
        els.append(Paragraph(f'<b>{i}.</b>  {_esc(q)}{tag}', st['li']))

    els.append(Spacer(1, 0.3 * cm))
    return els


def _build_articles(final: dict, sec_num: str) -> list:
    """Section 2 — Top Articles (ranked table + score breakdown)."""
    st     = _st()
    arts   = final.get('top_articles', [])
    scores = final.get('top_scores', [])
    els    = _section_header(f"Top Articles  ({len(arts)} selected)", sec_num)

    if not arts:
        els.append(Paragraph('No articles were selected by the Quality Critic.', st['body']))
        return els

    # ── Main ranked table ───────────────────────────────────────────────────
    hdr = [Paragraph(h, st['tc_b'])
           for h in ['#', 'Title', 'Venue', 'Year', 'Cited', 'Score']]
    rows = [hdr]
    for rank, (art, sc) in enumerate(zip(arts, scores), 1):
        # Truncate raw text BEFORE escaping and BEFORE adding any HTML tags.
        # Slicing after <font> tags are added would cut mid-tag → parser crash.
        venue_raw = (art.journal_name or '—')[:42]
        venue = _esc(venue_raw)
        if art.is_preprint:
            venue += ' <font size="7">(pp)</font>'
        t_short = _esc((art.title[:78] + '…') if len(art.title) > 78 else art.title)
        rows.append([
            Paragraph(str(rank), st['tc_c']),
            Paragraph(t_short,   st['tc']),
            Paragraph(venue, st['tc']),
            Paragraph(str(art.year), st['tc_c']),
            Paragraph(str(art.citation_count), st['tc_c']),
            Paragraph(f'<b>{sc.final_score:.2f}</b>', st['score']),
        ])

    cw1 = [CW * r for r in (0.05, 0.42, 0.24, 0.07, 0.09, 0.13)]
    t1  = Table(rows, colWidths=cw1, repeatRows=1)
    ts1 = _base_table_style(len(rows))
    ts1.add('ALIGN', (0, 0), (0, -1), 'CENTER')
    ts1.add('ALIGN', (3, 0), (5, -1), 'CENTER')
    t1.setStyle(ts1)
    els.append(t1)
    els.append(Spacer(1, 0.4 * cm))

    # ── Score breakdown sub-table ───────────────────────────────────────────
    els.append(Paragraph('<b>Score Breakdown</b> '
                         '<font size="8" color="#555555">'
                         '(venue / authors / impact / relevance)</font>',
                         st['sec_h2']))

    hdr2 = [Paragraph(h, st['tc_b'])
            for h in ['#', 'Title (short)', 'Venue', 'Authors', 'Impact', 'Relev.', 'Final']]
    rows2 = [hdr2]
    for rank, (art, sc) in enumerate(zip(arts, scores), 1):
        short = _esc((art.title[:42] + '...') if len(art.title) > 42 else art.title)
        rows2.append([
            Paragraph(str(rank), st['tc_c']),
            Paragraph(short,     st['tc']),
            Paragraph(f'{sc.venue_score:.2f}',     st['tc_c']),
            Paragraph(f'{sc.authors_score:.2f}',   st['tc_c']),
            Paragraph(f'{sc.impact_score:.2f}',    st['tc_c']),
            Paragraph(f'{sc.relevance_score:.2f}', st['tc_c']),
            Paragraph(f'<b>{sc.final_score:.2f}</b>', st['score']),
        ])

    cw2 = [CW * r for r in (0.05, 0.35, 0.12, 0.12, 0.12, 0.12, 0.12)]
    t2  = Table(rows2, colWidths=cw2, repeatRows=1)
    ts2 = _base_table_style(len(rows2))
    ts2.add('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ts2.add('ALIGN', (1, 0), (1, -1),  'LEFT')
    t2.setStyle(ts2)
    els.append(t2)
    return els


def _build_summaries(final: dict, sec_num: str) -> list:
    """Section 3 — Per-article summaries.

    Structured mode (default): 6-field key/value card per article.
    Narrative mode (--narrative): prose paragraph per article.
    """
    st        = _st()
    arts_map  = {a.id: a for a in final.get('top_articles', [])}
    narrative = final.get('narrative_mode', False)

    if narrative:
        summs = final.get('narrative_summaries', [])
    else:
        summs = final.get('summaries', [])

    mode_label = 'Narrative prose' if narrative else 'Structured 6-field'
    els = _section_header(
        f"Article Summaries  ({len(summs)} articles)  [{mode_label}]", sec_num
    )

    if not summs:
        els.append(Paragraph('No summaries were produced.', st['body']))
        return els

    for i, s in enumerate(summs, 1):
        art = arts_map.get(s.article_id)
        title_raw   = art.title if art else s.article_id
        title_short = _esc((title_raw[:88] + '…') if len(title_raw) > 88 else title_raw)
        venue = _esc((art.journal_name or '—')[:40]) if art else '—'
        year  = str(art.year) if art else '—'

        # ── Article title bar (shared by both modes) ─────────────────────────
        title_bar = Table(
            [[Paragraph(f'<b>[{i}]</b>  {title_short}', st['sec_h3']),
              Paragraph(f'{venue} &middot; {year}', st['tc_bs'])]],
            colWidths=[CW * 0.76, CW * 0.24],
        )
        title_bar.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), CBL),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        if narrative:
            # ── Narrative mode: single prose paragraph ────────────────────
            prose_wrap = Table(
                [[Paragraph(_esc(s.text or '—'), st['body'])]],
                colWidths=[CW],
            )
            prose_wrap.setStyle(TableStyle([
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND',    (0, 0), (-1, -1), CGL),
                ('BOX',           (0, 0), (-1, -1), 0.3, CGM),
            ]))
            els.append(KeepTogether([title_bar, prose_wrap, Spacer(1, 0.35 * cm)]))

        else:
            # ── Structured mode: 6-field key/value card ───────────────────
            card_rows: list[tuple[str, str]] = [
                ('Problem',  _esc(s.problem)),
                ('Method',   _esc(s.method)),
                ('Dataset',  _esc(s.dataset or '—')),
                ('Results',  _esc(s.results)),
            ]
            if s.limitations:
                card_rows.append(('Limitations', _esc(s.limitations)))
            if s.key_contributions:
                contribs = '  &#x2022;  '.join(_esc(c) for c in s.key_contributions)
                card_rows.append(('Contributions', contribs))

            card_data = [
                [Paragraph(f'<b>{k}</b>', st['tc_b']), Paragraph(v, st['tc'])]
                for k, v in card_rows
            ]
            card = Table(card_data, colWidths=[CW * 0.20, CW * 0.80])
            card_style_cmds = [
                ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING',    (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING',   (0, 0), (-1, -1), 5),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
                ('GRID',          (0, 0), (-1, -1), 0.3, CGM),
                ('BACKGROUND',    (0, 0), (0, -1),  CBL),
            ]
            for r in range(0, len(card_data), 2):
                card_style_cmds.append(('BACKGROUND', (1, r), (1, r), CGL))
            card.setStyle(TableStyle(card_style_cmds))
            els.append(KeepTogether([title_bar, card, Spacer(1, 0.35 * cm)]))

    return els


def _build_reflexion(final: dict, sec_num: str) -> list:
    """Section 4 — Reflexion loop: Critic feedback history."""
    st         = _st()
    feedbacks  = final.get('critic_feedbacks', [])
    n_iter     = final.get('synthesis_iteration', 1)
    els        = _section_header(
        f"Reflexion Loop  ({n_iter} iteration(s), {len(feedbacks)} critique(s))",
        sec_num,
    )

    # Config summary box
    config_info = (
        f"Config: <b>MAX_REFLEXION_ITERATIONS = {MAX_REFLEXION_ITERATIONS}</b>  "
        f"&#x2014;  "
        f"<b>REFLEXION_MIN_QUALITY = {REFLEXION_MIN_QUALITY.upper()}</b>.  "
        f"The Critic evaluates the Synthesis on 4 axes: <i>fidelity, completeness, "
        f"specificity, consistency</i>. If quality is below the threshold, "
        f"it requests a revision."
    )
    els.append(Paragraph(config_info, st['body']))
    els.append(Spacer(1, 0.2 * cm))

    if not feedbacks:
        els.append(Paragraph('<i>No Critic feedback was recorded.</i>', st['body_sm']))
        return els

    for fb in feedbacks:
        qcol   = QCOL.get(fb.overall_quality, CGD)
        apcol  = CGR if not fb.needs_revision else CRD
        ap_txt = 'APPROVED' if not fb.needs_revision else 'REVISION REQUESTED'

        iter_header = (
            f'<b>Iteration {fb.iteration}</b>'
            f'  &nbsp;&#x2014;&nbsp;  '
            f'Quality: {_colored(fb.overall_quality.upper(), qcol)}'
            f'  &nbsp;&#x2014;&nbsp;  '
            f'{_colored(ap_txt, apcol)}'
        )
        els.append(Paragraph(iter_header, st['sec_h2']))

        if fb.issues:
            els.append(Paragraph('<b>Issues identified by Critic:</b>', st['sec_h3']))
            for issue in fb.issues:
                els.append(_li(_esc(issue)))
        if fb.suggestions:
            els.append(Paragraph('<b>Suggestions given to Synthesizer:</b>',
                                 st['sec_h3']))
            for sug in fb.suggestions:
                els.append(_li(f'<i>{_esc(sug)}</i>'))
        if not fb.issues and not fb.suggestions:
            els.append(Paragraph(
                '<i>No issues identified &#x2014; synthesis accepted as-is.</i>',
                st['body_sm'],
            ))

        els.append(Spacer(1, 0.1 * cm))
        els.append(_hr())

    els.append(Spacer(1, 0.15 * cm))
    els.append(Paragraph(
        f'<b>Final:</b> Synthesizer completed <b>{n_iter}</b> iteration(s).  '
        f'Maximum allowed: <b>{MAX_REFLEXION_ITERATIONS}</b>.  '
        f'Quality threshold required: '
        f'{_colored(REFLEXION_MIN_QUALITY.upper(), QCOL.get(REFLEXION_MIN_QUALITY, CGD))}.',
        st['body'],
    ))
    return els


def _build_synthesis(final: dict, sec_num: str) -> list:
    """Section 5 — Global Synthesis."""
    st     = _st()
    synth  = final.get('synthesis')
    n_iter = final.get('synthesis_iteration', 1)
    els    = _section_header(
        f"Global Synthesis  (after {n_iter} iteration(s))", sec_num
    )

    if not synth:
        els.append(Paragraph('No synthesis was produced.', st['body']))
        return els

    els.append(Paragraph('<b>Overview</b>', st['sec_h2']))
    els.append(Paragraph(_esc(synth.overview), st['body']))
    els.append(Spacer(1, 0.2 * cm))

    if synth.main_approaches:
        els.append(Paragraph('<b>Main Approaches</b>', st['sec_h2']))
        for a in synth.main_approaches:
            els.append(_li(_esc(a)))
        els.append(Spacer(1, 0.2 * cm))

    if synth.common_datasets:
        els.append(Paragraph('<b>Common Datasets</b>', st['sec_h2']))
        els.append(Paragraph(
            '  &#x2022;  '.join(f'<b>{_esc(d)}</b>' for d in synth.common_datasets),
            st['body'],
        ))
        els.append(Spacer(1, 0.2 * cm))

    if synth.key_findings:
        els.append(Paragraph('<b>Key Findings</b>', st['sec_h2']))
        for j, finding in enumerate(synth.key_findings, 1):
            els.append(Paragraph(f'<b>{j}.</b>  {_esc(finding)}', st['li']))
        els.append(Spacer(1, 0.2 * cm))

    els.append(Paragraph(
        f'<i>Synthesis covers <b>{synth.article_count}</b> article(s).</i>',
        st['body_sm'],
    ))
    return els


def _build_trends(final: dict, sec_num: str) -> list:
    """Section 1 (new order) — Trends, Research Gaps & Future Perspectives.

    For EMERGING trends, also shows the supporting article titles with a
    2-3 sentence digest (problem + method) so the reader immediately
    understands *what work* underpins each emerging direction.
    """
    st  = _st()
    ta  = final.get('trend_analysis')
    els = _section_header('Trends, Gaps & Future Perspectives', sec_num)

    if not ta:
        els.append(Paragraph('No trend analysis was produced.', st['body']))
        return els

    # Build fast lookup dicts so we can enrich emerging trend rows
    # with article titles and summaries (structured or narrative).
    narrative = final.get('narrative_mode', False)
    art_by_id = {a.id: a for a in final.get('top_articles', [])}
    _summ_list = (
        final.get('narrative_summaries', [])
        if narrative
        else final.get('summaries', [])
    )
    sum_by_id = {s.article_id: s for s in _summ_list}

    # ── Research Trends table ─────────────────────────────────────────────────
    if ta.trends:
        els.append(Paragraph('<b>Research Trends</b>', st['sec_h2']))
        tr_hdr  = [Paragraph(h, st['tc_b'])
                   for h in ['Trend', 'Maturity', 'Description', 'Evidence']]
        tr_rows = [tr_hdr]
        for tr in ta.trends:
            mc  = MCOL.get(tr.maturity, CGD)
            ev  = _esc(', '.join(tr.evidence_article_ids[:3])) or '—'
            tr_rows.append([
                Paragraph(f'<b>{_esc(tr.name)}</b>', st['tc']),
                Paragraph(_colored(tr.maturity.upper(), mc), st['tc_c']),
                Paragraph(_esc(tr.description), st['tc']),
                Paragraph(ev, st['tc_bs']),
            ])
        cw_tr = [CW * r for r in (0.23, 0.14, 0.43, 0.20)]
        tt = Table(tr_rows, colWidths=cw_tr, repeatRows=1)
        tt.setStyle(_base_table_style(len(tr_rows)))
        els.append(tt)
        els.append(Spacer(1, 0.3 * cm))

    # ── Emerging Trends — Supporting Articles ─────────────────────────────────
    # Only EMERGING trends get this expanded view.  For each one we list the
    # supporting article titles and a 2-3 sentence digest (problem + method)
    # so the reader knows *which papers* drive this emerging direction.
    emerging = [tr for tr in (ta.trends or []) if tr.maturity == 'emerging']
    if emerging:
        els.append(Paragraph(
            '<b>Emerging Trends — Supporting Articles</b>', st['sec_h2']
        ))
        for tr in emerging:
            # Trend header with green badge
            badge = _colored('[EMERGING]', CGR)
            els.append(Paragraph(
                f'{badge}  <b>{_esc(tr.name)}</b>', st['sec_h3']
            ))
            els.append(Spacer(1, 0.05 * cm))

            shown = 0
            for aid in tr.evidence_article_ids[:4]:
                art  = art_by_id.get(aid)
                summ = sum_by_id.get(aid)
                if not art and not summ:
                    continue  # article not in top list — skip

                # Article title + year on one line
                title = _esc(art.title) if art else _esc(aid)
                year  = f' ({art.year})' if art else ''
                els.append(Paragraph(
                    f'<b>{title}{year}</b>', st['body']
                ))

                # 2-3 sentence digest — adapts to summary format:
                #   Structured: problem + method + first result sentence
                #   Narrative : first 2 sentences of the prose text
                _skip = {'not specified', ''}
                digest_parts: list[str] = []
                if summ:
                    if hasattr(summ, 'text'):
                        # NarrativeSummary — extract first 2 sentences
                        sentences = [
                            s.strip() for s in (summ.text or '').split('.')
                            if s.strip() and len(s.strip()) > 20
                        ]
                        digest_parts = [_esc(s + '.') for s in sentences[:2]]
                    else:
                        # ArticleSummary — problem + method + first result sentence
                        if (summ.problem or '').lower().strip() not in _skip:
                            digest_parts.append(_esc(summ.problem))
                        if (summ.method or '').lower().strip() not in _skip:
                            digest_parts.append(_esc(summ.method))
                        if (summ.results or '').lower().strip() not in _skip:
                            first_result = summ.results.split('.')[0].strip()
                            if len(first_result) > 20:
                                digest_parts.append(_esc(first_result + '.'))

                if digest_parts:
                    els.append(Paragraph('  '.join(digest_parts), st['body_sm']))
                els.append(Spacer(1, 0.12 * cm))
                shown += 1

            if shown == 0:
                els.append(Paragraph(
                    '<i>No article details available for this trend.</i>',
                    st['body_sm'],
                ))
            els.append(Spacer(1, 0.2 * cm))

    # ── Research Gaps ─────────────────────────────────────────────────────────
    if ta.gaps:
        els.append(Paragraph('<b>Research Gaps</b>', st['sec_h2']))
        for g in ta.gaps:
            ic    = ICOL.get(g.importance, CGD)
            badge = _colored(f'[{g.importance.upper()}]', ic)
            els.append(KeepTogether([
                Paragraph(f'{badge}  {_esc(g.description)}', st['li']),
                *[_sub_li(_esc(d)) for d in g.suggested_directions[:2]],
                Spacer(1, 0.1 * cm),
            ]))
        els.append(Spacer(1, 0.2 * cm))

    # ── Future Perspectives ───────────────────────────────────────────────────
    if ta.future_perspectives:
        els.append(Paragraph('<b>Future Perspectives</b>', st['sec_h2']))
        for p in ta.future_perspectives:
            els.append(_li(_esc(p)))

    return els


def _build_logs(final: dict, sec_num: str) -> list:
    """Section 7 — Agent Execution Logs + Errors."""
    st     = _st()
    logs   = final.get('logs', [])
    errors = final.get('errors', [])
    total  = sum(lg.tokens_used for lg in logs)
    els    = _section_header(f"Agent Execution Logs  ({len(logs)} agents)", sec_num)

    if logs:
        lg_hdr  = [Paragraph(h, st['tc_b'])
                   for h in ['Agent', 'Status', 'Duration (s)', 'API Calls', 'Tokens']]
        lg_rows = [lg_hdr]
        for lg in logs:
            dur   = (
                (lg.completed_at - lg.started_at).total_seconds()
                if lg.completed_at else 0
            )
            ok    = lg.status == 'success'
            scol  = CGR if ok else CRD
            badge = _colored('OK' if ok else 'ERR', scol)
            lg_rows.append([
                Paragraph(_esc(lg.agent_name), st['tc']),
                Paragraph(badge,              st['tc_c']),
                Paragraph(f'{dur:.2f}',       st['tc_c']),
                Paragraph(str(lg.api_calls),  st['tc_c']),
                Paragraph(f'{lg.tokens_used:,}', st['tc_c']),
            ])
        # Total row
        lg_rows.append([
            Paragraph('<b>TOTAL</b>',      st['tc_b']),
            Paragraph('&#x2014;',          st['tc_c']),
            Paragraph('&#x2014;',          st['tc_c']),
            Paragraph('&#x2014;',          st['tc_c']),
            Paragraph(f'<b>{total:,}</b>', st['score']),
        ])

        cw_lg = [CW * r for r in (0.32, 0.12, 0.18, 0.15, 0.23)]
        lt    = Table(lg_rows, colWidths=cw_lg, repeatRows=1)
        ts    = _base_table_style(len(lg_rows), has_total_row=True)
        ts.add('ALIGN', (1, 0), (4, -1), 'CENTER')
        lt.setStyle(ts)
        els.append(lt)

        # Per-agent errors
        agent_errors = [(lg.agent_name, lg.error) for lg in logs if lg.error]
        if agent_errors:
            els.append(Spacer(1, 0.2 * cm))
            els.append(Paragraph('<b>Agent-level errors:</b>', st['sec_h3']))
            for name, err in agent_errors:
                els.append(_li(
                    f'<font color="{_hex(CRD)}"><b>{_esc(name)}</b></font>: '
                    f'{_esc(err[:200])}'
                ))

    # Pipeline-level errors
    if errors:
        els.append(Spacer(1, 0.25 * cm))
        els.append(Paragraph('<b>Pipeline errors:</b>', st['sec_h2']))
        for err in errors:
            els.append(_li(
                f'<font color="{_hex(CRD)}">{_esc(err[:300])}</font>'
            ))

    return els


# ── Public API ────────────────────────────────────────────────────────────────

def export_to_pdf(
    final: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """Generate a PDF report from a pipeline result dict.

    Args:
        final:       The dict returned by run_pipeline().
        output_path: Destination .pdf path. If None, auto-generates
                     'report_<safe_topic>_<YYYYMMDD_HHMMSS>.pdf'
                     in the current working directory.

    Returns:
        Absolute path of the generated PDF file.
    """
    # ── Resolve output path ────────────────────────────────────────────────
    if output_path is None:
        topic = final.get('topic', 'report')
        safe  = ''.join(c if (c.isalnum() or c in '_-') else '_' for c in topic)
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'report_{safe}_{ts}.pdf'
    output_path = str(Path(output_path).resolve())

    topic = final.get('topic', 'Unknown Topic')
    logger.info("Building PDF report for topic=%r -> %s", topic, output_path)

    # ── Page header/footer (closure captures topic) ────────────────────────
    def _page_frame(canvas, doc):
        canvas.saveState()
        # Bottom footer
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(CGD)
        footer_txt = (
            f'Scientific Watch Report agent ReAct+Relexion --  {topic}  --  Page {doc.page}'
        )
        canvas.drawCentredString(_PW / 2, 0.75 * cm, footer_txt)
        # Top separator line
        canvas.setStrokeColor(CGM)
        canvas.setLineWidth(0.4)
        canvas.line(_LR, _PH - _TB + 0.15 * cm,
                    _PW - _LR, _PH - _TB + 0.15 * cm)
        canvas.restoreState()

    # ── Build document ─────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=_LR,
        rightMargin=_LR,
        topMargin=_TB + 0.3 * cm,
        bottomMargin=_TB,
        title=f'Scientific Watch Report agent ReAct+Reflexion — {topic}',
        author='Scientific Watch Report agent ReAct+Reflexion',
    )

    story: list = []
    story += _build_cover(final)
    # Section order: insights first (trends → synthesis → articles),
    # then methodology details (queries → reflexion → logs).
    story += _build_trends(final, '1')
    story.append(PageBreak())
    story += _build_synthesis(final, '2')
    story.append(PageBreak())
    story += _build_articles(final, '3')
    story.append(PageBreak())
    story += _build_summaries(final, '4')
    story.append(PageBreak())
    story += _build_queries(final, '5')
    story.append(PageBreak())
    story += _build_reflexion(final, '6')
    story.append(PageBreak())
    story += _build_logs(final, '7')

    doc.build(story, onFirstPage=_page_frame, onLaterPages=_page_frame)
    logger.info("PDF report saved: %s", output_path)
    return output_path


# ── Email delivery ───────────────────────────────────────────────────────────

def send_report_by_email(
    pdf_path: str,
    topic: str,
    recipient: str,
    final: dict[str, Any] | None = None,
    sender: str | None = None,
    password: str | None = None,
) -> None:
    """Send the generated PDF report as an email attachment via Gmail SMTP.

    Uses STARTTLS on port 587 (the recommended secure method for Gmail).
    Requires a Gmail App Password — NOT your regular Gmail password:
        Google Account → Security → 2-Step Verification → App passwords
        Select "Mail" → generate → copy the 16-char password to .env.

    Args:
        pdf_path:  Absolute path to the PDF file to attach.
        topic:     Research topic (shown in subject + body).
        recipient: Destination email address.
        final:     Pipeline result dict (used to build a rich email body).
                   If None, a minimal body is sent.
        sender:    Gmail address.  Falls back to config.EMAIL_SENDER.
        password:  Gmail App Password.  Falls back to config.EMAIL_PASSWORD.

    Raises:
        ValueError: if sender or password are not configured.
        smtplib.SMTPException: on SMTP-level errors (auth failure, etc.).
    """
    sender   = sender   or EMAIL_SENDER
    password = password or EMAIL_PASSWORD

    if not sender or not password:
        raise ValueError(
            "Email credentials not configured.\n"
            "Set EMAIL_SENDER and EMAIL_PASSWORD in your .env file.\n"
            "Gmail requires an App Password (not your regular password):\n"
            "  Google Account → Security → 2-Step Verification → App passwords"
        )

    # ── Build email body ──────────────────────────────────────────────────
    now = datetime.now().strftime('%d %B %Y at %H:%M')

    if final:
        cfg          = final.get('config', {})
        n_iter       = final.get('synthesis_iteration', 1)
        last_fb      = (final.get('critic_feedbacks') or [None])[-1]
        final_qual   = last_fb.overall_quality.upper() if last_fb else 'N/A'
        n_articles   = len(final.get('top_articles', []))
        n_summaries  = len(final.get('summaries', []))
        total_tokens = sum(lg.tokens_used for lg in final.get('logs', []))

        body_lines = [
            f'Scientific Watch Agent — Report Ready',
            f'',
            f'Topic       : "{topic}"',
            f'Generated   : {now}',
            f'',
            f'Pipeline summary',
            f'  Articles fetched    : {cfg.get("n_raw", "?")}',
            f'  Articles selected   : {n_articles}',
            f'  Summaries produced  : {n_summaries}',
            f'  Reflexion iterations: {n_iter} / {MAX_REFLEXION_ITERATIONS}',
            f'  Final quality       : {final_qual}',
            f'  Total tokens used   : {total_tokens:,}',
            f'',
            f'The full report is attached as a PDF.',
            f'',
            f'---',
            f'Scientific Watch Agent — automated research monitoring',
        ]
    else:
        body_lines = [
            f'Scientific Watch Agent — Report Ready',
            f'',
            f'Topic     : "{topic}"',
            f'Generated : {now}',
            f'',
            f'The full report is attached as a PDF.',
            f'',
            f'---',
            f'Scientific Watch Agent — automated research monitoring',
        ]

    body = '\n'.join(body_lines)

    # ── Assemble MIME message ─────────────────────────────────────────────
    msg = MIMEMultipart()
    msg['From']    = sender
    msg['To']      = recipient
    msg['Subject'] = f'[Scientific Watch] Report — "{topic}" — {now}'
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # Attach the PDF
    pdf_bytes = Path(pdf_path).read_bytes()
    attachment = MIMEApplication(pdf_bytes, _subtype='pdf')
    attachment.add_header(
        'Content-Disposition',
        'attachment',
        filename=Path(pdf_path).name,
    )
    msg.attach(attachment)

    # ── Send via Gmail SMTP (STARTTLS) ────────────────────────────────────
    logger.info("Connecting to %s:%d ...", SMTP_HOST, SMTP_PORT)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, password)
        smtp.sendmail(sender, recipient, msg.as_string())

    logger.info("Report emailed to %s", recipient)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Run the Scientific Watch pipeline and export results to PDF.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'topic', nargs='?', default='fake news detection',
        help='Research topic to investigate (default: "fake news detection")',
    )
    parser.add_argument(
        'n_raw', nargs='?', type=int, default=30,
        help='Number of articles to fetch from OpenAlex (default: 30)',
    )
    parser.add_argument(
        'top_n', nargs='?', type=int, default=5,
        help='Number of top articles to keep after quality filtering (default: 5)',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output PDF file path (auto-generated if omitted)',
    )
    parser.add_argument(
        '--email', '-e', default=None, metavar='ADDRESS',
        help=(
            'Send the PDF report to this email address after generation.\n'
            'Requires EMAIL_SENDER and EMAIL_PASSWORD in your .env file.\n'
            'Gmail: use an App Password, not your regular password.\n'
            '  Google Account → Security → 2-Step Verification → App passwords'
        ),
    )
    parser.add_argument(
        '--narrative', action='store_true',
        help=(
            'Use narrative summary mode: summaries are prose paragraphs '
            '(150-250 words) instead of the default 6-field structured cards. '
            'The PDF Section 4 renders each summary as a flowing paragraph.'
        ),
    )
    args = parser.parse_args()

    mode_label = 'NARRATIVE' if args.narrative else 'STRUCTURED'
    print(f"Running pipeline  |  topic='{args.topic}'  n_raw={args.n_raw}  top_n={args.top_n}  mode={mode_label}")
    final = run_pipeline(
        args.topic,
        n_raw=args.n_raw,
        top_n=args.top_n,
        narrative_mode=args.narrative,
    )

    print("Generating PDF report ...")
    path = export_to_pdf(final, output_path=args.output)
    print(f"Report saved: {path}")

    # ── Optional: email delivery ──────────────────────────────────────────
    recipient = args.email or EMAIL_RECEIVER
    if recipient:
        print(f"Sending report to {recipient} ...")
        try:
            send_report_by_email(
                pdf_path=path,
                topic=args.topic,
                recipient=recipient,
                final=final,
            )
            print(f"✓ Report sent to {recipient}")
        except ValueError as exc:
            print(f"\n[Email config error]\n{exc}\n")
            print("The PDF was saved locally. Configure .env to enable email delivery.")
        except Exception as exc:
            print(f"\n[Email send failed] {exc}")
            print(f"The PDF was saved locally at: {path}")


if __name__ == '__main__':
    main()
