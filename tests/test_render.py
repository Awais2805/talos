"""AuditPage's embedded payload must survive a real browser's <script> parsing.

A browser never decodes HTML entities inside <script>...</script> -- it is a
RAWTEXT element per the HTML5 spec. html.escape() assumes the opposite (that
whatever reads its output will entity-decode), which is right for ordinary
HTML text and wrong here. These tests read the embedded text exactly as
`.textContent` would hand it to JSON.parse: no html.unescape() first.
"""

from __future__ import annotations

import json
import re

import duckdb
import pytest

from talos.common.duck import DuckEngine
from talos.data.labelling.audit import AuditPage
from talos.data.labelling.space import LabelSpaceLoader


@pytest.fixture
def csv_with_special_chars(tmp_path):
    """A candidate row containing the characters that broke this before:
    a quote, an ampersand, and a `<` (the one genuinely dangerous one)."""
    path = tmp_path / "candidates.csv"
    path.write_text(
        'uid,ts,capture,dataset,note,prior_method,prior_class,prior_source,'
        'prior_quality,oracle_alerted,oracle_signature,analyst_class,analyst_note\n'
        'u1,1.0,cap,toy,"quote"" amp & lt<",schedule,benign,matched,certain,'
        ',,,\n'
    )
    return path


def embedded_text(html_path):
    """Exactly what `.textContent` on the <script> tag would return -- no
    entity decoding, because a browser never does that inside RAWTEXT."""
    return re.search(r'type="application/json">(.*?)</script>',
                     html_path.read_text(), re.S).group(1)


def test_the_payload_parses_as_json_without_entity_decoding(csv_with_special_chars, tmp_path):
    """The regression: html.escape() left no real quote characters at all,
    so JSON.parse threw before the page ever built a table -- a blank page
    with just the button, exactly as it looked to the person adjudicating."""
    space = LabelSpaceLoader().load("core-5")
    duck = DuckEngine()
    html_path = AuditPage(space, "toy").render(
        duck, csv_with_special_chars, tmp_path / "page.html")

    raw = embedded_text(html_path)
    data = json.loads(raw)          # must not raise
    assert data["rows"][0]["note"] == 'quote" amp & lt<'


def test_no_literal_angle_bracket_can_close_the_script_tag_early(
        csv_with_special_chars, tmp_path):
    """The one character RAWTEXT parsing actually cares about: a literal `<`
    in the payload could spell `</script>` and truncate the page before the
    JSON even ends. `<` is a legal JSON escape for the same character, so
    encoding it costs nothing JSON.parse can't undo."""
    space = LabelSpaceLoader().load("core-5")
    duck = DuckEngine()
    html_path = AuditPage(space, "toy").render(
        duck, csv_with_special_chars, tmp_path / "page.html")

    assert "<" not in embedded_text(html_path)


def test_the_context_panel_survives_the_same_script_tag_parsing(
        csv_with_special_chars, tmp_path):
    """The context feature is new data flowing through the same <script>
    embedding -- it must clear the same bar the row payload does, not get a
    free pass because it arrived after the escaping bug was already fixed."""
    space = LabelSpaceLoader().load("core-5")
    duck = DuckEngine()
    context = {"u1": [{"uid": "n1", "ts": 2.0, "orig": "10.0.0.1",
                       "resp": "10.0.0.2", "conn_state": "S0<>&\""}]}
    html_path = AuditPage(space, "toy").render(
        duck, csv_with_special_chars, tmp_path / "page.html", context)

    raw = embedded_text(html_path)
    data = json.loads(raw)
    assert data["context"]["u1"][0]["conn_state"] == 'S0<>&"'
    assert "<" not in raw
