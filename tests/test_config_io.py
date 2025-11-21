import io

import pytest

from config_io import (
    build_fund_type_map,
    build_label_map,
    load_config,
    parse_fund_id_from_label,
)


def test_parse_fund_id_from_label_extracts_id():
    label = "AMF Räntefond Lång (2111)"
    assert parse_fund_id_from_label(label) == "2111"


def test_parse_fund_id_from_label_invalid_returns_none():
    assert parse_fund_id_from_label("AMF Räntefond Lång") is None


def test_load_config_requires_list():
    bad_config = io.StringIO('{"fund_id": "1"}')
    with pytest.raises(ValueError):
        load_config(bad_config)


def test_build_maps_extract_values():
    config = [
        {"fund_id": "2111", "type": "Räntefond", "label": "AMF Räntefond Lång (2111)"},
        {"fund_id": "1234", "type": "Aktiefond", "label": "Aktiefond (1234)"},
    ]

    assert build_fund_type_map(config) == {"2111": "Räntefond", "1234": "Aktiefond"}
    assert build_label_map(config) == {
        "2111": "AMF Räntefond Lång (2111)",
        "1234": "Aktiefond (1234)",
    }
