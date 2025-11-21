import json
from typing import List, Dict, Optional


def parse_fund_id_from_label(label: str) -> Optional[str]:
    """
    Försöker plocka ut fond-ID från namn av typen 'Fondnamn (12345)'.
    """
    if "(" in label and label.endswith(")"):
        return label.split("(")[-1].strip(")")
    return None


def load_config(file_obj) -> List[Dict]:
    """
    Läser en JSON-konfig med format:
    [
      {"fund_id": "2111", "type": "Räntefond", "label": "AMF Räntefond Lång (2111)"},
      ...
    ]
    """
    data = json.load(file_obj)
    if not isinstance(data, list):
        raise ValueError("Konfigurationsfilen ska vara en lista av objekt.")
    return data


def build_fund_type_map(config_list: List[Dict]) -> Dict[str, str]:
    """
    Bygger en mapping {fund_id: typ} från konfigurationslistan.
    """
    fund_type_map = {}
    for item in config_list:
        fid = str(item.get("fund_id", "")).strip()
        ftype = item.get("type")
        if fid and ftype:
            fund_type_map[fid] = ftype
    return fund_type_map


def build_label_map(config_list: List[Dict]) -> Dict[str, str]:
    """
    Bygger mapping {fund_id: label} om du vill nyttja sparade labels.
    """
    label_map = {}
    for item in config_list:
        fid = str(item.get("fund_id", "")).strip()
        lab = item.get("label")
        if fid and lab:
            label_map[fid] = lab
    return label_map


def create_config(
    selected_names,
    fund_types: Dict[str, str],
) -> str:
    """
    Skapar en JSON-sträng med konfiguration baserat på valda fonder
    och deras klassificering.
    """
    config = []
    for name in selected_names:
        ftype = fund_types.get(name)
        fund_id = parse_fund_id_from_label(name)
        config.append(
            {
                "fund_id": fund_id,
                "type": ftype,
                "label": name,
            }
        )

    return json.dumps(config, ensure_ascii=False, indent=2)
