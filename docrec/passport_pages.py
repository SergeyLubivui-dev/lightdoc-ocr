"""Несколько разворотов паспорта на одном листе: классификация каждого куска (основной разворот / страница регистрации / другая страница паспорта),
полное чтение основного разворота (passport.read_passport), best-effort чтение страницы регистрации (штампы, орган, серия/номер) и сверка серии/номера между страницами."""
from __future__ import annotations
import re
from collections import Counter
import cv2
import numpy as np
from .doctypes.passport import cue_count, find_dates, valid_date
from .textutils import partial_ratio
from . import lexicon as LX

REG_CUES = [r"ЗАРЕГИСТР", r"РЕГИСТРАЦИ", r"МЕСТО\s*ЖИТЕЛЬСТВА", r"УФМС", r"ОТДЕЛ\b", r"СНЯТ\s*С", r"ПРОПИС", r"ПОДПИСЬ\s*(?:ДОЛЖНОСТНОГО|СОТРУДНИКА)?"]
STRONG = [r"ЗАРЕГИСТР", r"МЕСТО\s*ЖИТЕЛЬСТВА", r"ОТДЕЛ\s*УФМС", r"УФМС\s*РОССИИ", r"РЕГИСТРАЦИОННОГО"]
SN_RE = re.compile(r"(?<!\d)(\d{2})\s*(\d{2})\s*(\d{6})(?!\d)")
ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}


def _norm_up(t: str) -> str:
    return t.upper().replace("Ё", "Е")


def _fuzzy_cue(text: str, pat: str) -> bool:
    """«ЗАРЕГИСТРИРОЕАН», «Тверскои»: допускаем 1–2 ошибки OCR для длинных ключевых слов."""
    if re.search(pat, text): return True
    word = {r"ЗАРЕГИСТР": "ЗАРЕГИСТРИРОВАН", r"РЕГИСТРАЦИОННОГО": "РЕГИСТРАЦИОННОГО", r"УФМС\s*РОССИИ": "УФМС РОССИИ", r"ОТДЕЛ\s*УФМС": "ОТДЕЛ УФМС", r"МЕСТО\s*ЖИТЕЛЬСТВА": "МЕСТО ЖИТЕЛЬСТВА"}.get(pat)
    return bool(word) and partial_ratio(word, text) >= 0.82 and len(text) <= len(word) + 8


def _stamp_list(passes, hits):
    """Штампы по ориентации с наибольшим числом отметок: тип (ЗАРЕГИСТРИРОВАН / СНЯТ С РЕГИСТРАЦИОННОГО УЧЕТА), печатный орган из строк выше отметки,
    дата рядом и рукописные строки (best-effort, низкая уверенность — рукописный текст не доверяем)."""
    marks = [(rot, kind, l) for rot, p, l in hits for kind in [("registered" if p == r"ЗАРЕГИСТР" else "deregistered" if p == r"СНЯТ\s*С" else None)] if kind]
    if not marks: return []
    cnt = Counter(rot for rot, _, _ in marks); rot_b = max(cnt, key=lambda r: (cnt[r], r == 0))
    ms = sorted([(k, l) for r, k, l in marks if r == rot_b], key=lambda x: (x[1].cy, x[1].x0))
    uniq = []
    for k, l in ms:       # одна отметка на штамп: сливаем строки одного типа, лежащие вплотную
        if uniq and uniq[-1][0] == k and abs(uniq[-1][1].cy - l.cy) < 0.03 and abs(uniq[-1][1].x0 - l.x0) < 0.2: continue
        uniq.append((k, l))
    ls = sorted(passes[rot_b], key=lambda l: (l.cy, l.x0)); out = []
    HDR = re.compile(r"МЕСТО\s*ЖИТЕЛЬСТВА|ПОДПИСЬ|ВОИНСК|СЕМЕЙН|^\W*ДЕТИ\W*$")
    tops = []
    for i, (k, m) in enumerate(uniq):
        prev_y = uniq[i - 1][1].cy if i else max(0.0, m.cy - 0.22)
        tops.append([l for l in ls if max(prev_y + 0.01, m.cy - 0.2) < l.cy < m.cy - 0.003 and abs(l.x0 - m.x0) < 0.25 and not re.search(r"\d{2}\.\d{2}\.\d{2,4}", l.text) and not HDR.search(_norm_up(l.text))])
    for i, (k, m) in enumerate(uniq):
        prev_y = uniq[i - 1][1].cy if i else max(0.0, m.cy - 0.22); nxt_y = uniq[i + 1][1].cy if i + 1 < len(uniq) else min(1.0, m.cy + 0.22)
        top = tops[i]
        auth_raw = " ".join(l.text for l in top[-3:]); auth = None; ch = []
        if re.search(r"УФМС|ОТДЕЛ|ПО\s|ОБЛ|РЕСП|РАЙОН", _norm_up(auth_raw)):
            auth, ch = LX.snap_multi(auth_raw, [("passport/issuer_places", 0.1, 7), ("passport/issuer_words", 0.15, 4)])
        nxt_ids = {id(l) for l in (tops[i + 1] if i + 1 < len(tops) else [])}
        below = [l for l in ls if m.cy + 0.003 < l.cy < nxt_y - 0.01 and abs(l.x0 - m.x0) < 0.3 and id(l) not in nxt_ids and not HDR.search(_norm_up(l.text))]
        dates = [d for l in below + [m] for d, _ in find_dates(l.text) if valid_date(d, 1992)]
        hand = [l for l in below if not re.search(r"УФМС|ОТДЕЛ|РЕГИСТРАЦ|ЗАРЕГ|СНЯТ|ПОДПИСЬ|М\.П", _norm_up(l.text)) and re.search(r"[A-Za-zА-Яа-я0-9]{3}", l.text)]
        out.append(dict(type=k, type_text=("ЗАРЕГИСТРИРОВАН" if k == "registered" else "СНЯТ С РЕГИСТРАЦИОННОГО УЧЕТА"), confidence=round(float(m.score), 3),
                        authority=dict(value=auth, confidence=round(float(np.mean([l.score for l in top[-3:]])) * 0.9, 3) if auth else 0.0, raw=auth_raw, lexicon=[f"{a}→{b}" for a, b, _ in ch]) if auth else None,
                        date=dates[0] if dates else None,
                        handwriting=dict(lines=[l.text for l in hand[:6]], confidence=round(float(np.mean([l.score for l in hand])) * 0.3, 3), note="рукописные записи — best-effort, не доверять") if hand else None))
    return out


def read_registration(pipe, crop: np.ndarray) -> dict:
    """Кусок без признаков основного разворота: ищем штампы регистрации/выписки (в т.ч. повёрнутые на 90°), серию/номер и орган регистрации.
    -> dict(kind, cues, stamps, fields{...}, rotation, lines_n)."""
    paddle = pipe.paddle
    H0, W0 = crop.shape[:2]; mx = 1500
    s = mx / max(H0, W0) if max(H0, W0) > mx * 1.2 else 1.0
    img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else crop
    passes = {}
    for rot in (0, 90, 270):
        im = img if ROT[rot] is None else cv2.rotate(img, ROT[rot])
        passes[rot] = [l for l in paddle(im, merge=False) if l.score >= 0.4]
    cues = {}; hits = []
    for rot, ls in passes.items():
        for l in ls:
            t = _norm_up(l.text)
            for pat in STRONG + [r"УФМС", r"ОТДЕЛ\b", r"СНЯТ\s*С"]:
                if _fuzzy_cue(t, pat): hits.append((rot, pat, l)); cues[pat] = cues.get(pat, 0) + 1
    strong = sum(1 for p in STRONG if cues.get(p))
    # серия/номер: любая ориентация (вертикальная полоса у края)
    sn = None
    for rot, ls in passes.items():
        for l in ls:
            m = SN_RE.search(re.sub(r"[^\d ]", "", l.text)) or SN_RE.search(re.sub(r"\D", "", l.text))
            if m:
                cand = dict(series=f"{m.group(1)} {m.group(2)}", number=m.group(3), score=float(l.score), rotation=rot)
                if sn is None or cand["score"] > sn["score"]: sn = cand
    # штампы: «ЗАРЕГИСТРИРОВАН/СНЯТ» — одна отметка на штамп; шапка «Отдел УФМС России по …» — по одной на штамп; берём максимум из оценок
    from collections import Counter
    c_reg = Counter(rot for rot, p, _ in hits if p == r"ЗАРЕГИСТР"); c_auth = Counter(rot for rot, _id in {(rot, id(l)) for rot, p, l in hits if p in (r"ОТДЕЛ\s*УФМС", r"УФМС\s*РОССИИ")} for rot in [rot])
    n_reg = max(c_reg.values(), default=0); n_auth = max(c_auth.values(), default=0)      # один штамп виден в нескольких ориентациях — не суммируем
    stamps = max(n_reg, n_auth, 1 if strong else 0)
    # орган регистрации из шапки штампа (текст берём из ориентации, где шапка прочиталась лучше всего)
    authority = None; auth_conf = 0.0; fields = {}
    best = None
    for rot, p, l in hits:
        if p in (r"ОТДЕЛ\s*УФМС", r"УФМС\s*РОССИИ") and (best is None or l.score > best[2].score): best = (rot, p, l)
    if best:
        rot, _, l0 = best; ls = sorted(passes[rot], key=lambda l: (l.cy, l.x0))
        group = [l0]
        for l in ls:        # строки штампа идут подряд сверху вниз с близким левым краем
            if l.cy <= group[-1].cy + 0.004 or l.cy - group[-1].cy > 0.05 or abs(l.x0 - l0.x0) > 0.12 or re.search(r"ЗАРЕГ|\d{2}\.\d{2}", _norm_up(l.text)): continue
            group.append(l)
            if len(group) >= 3: break
        raw = " ".join(l.text for l in sorted(group, key=lambda l: l.cy))
        txt, ch = LX.snap_multi(raw, [("passport/issuer_places", 0.1, 7), ("passport/issuer_words", 0.15, 4)])
        authority = txt; auth_conf = float(np.mean([l.score for l in group])) * 0.9
        note = "штамп на странице регистрации (best-effort)" + ("; лексикон: " + ", ".join(f"«{a}» → «{b}» ({n})" for a, b, n in ch) if ch else "")
        fields["registration_authority"] = dict(value=txt, confidence=round(auth_conf, 3), note=note, raw=raw)
    # год/дата в штампе (рукописный/плохой оттиск): только валидные даты, иначе — годы как подсказка
    dates = []
    for rot, ls in passes.items():
        for l in ls:
            for d, _ in find_dates(l.text):
                if valid_date(d, 1992): dates.append((d, float(l.score)))
    if dates:
        d, sc = max(dates, key=lambda x: x[1]); fields["registration_date"] = dict(value=d, confidence=round(sc * 0.8, 3), note="дата в штампе (best-effort)")
    else:
        years = sorted({m.group(0) for _, ls in passes.items() for l in ls for m in re.finditer(r"(?<!\d)(?:19[9]\d|20[0-3]\d)(?!\d)", l.text) if l.score >= 0.7})
        if years: fields["registration_date"] = dict(value=None, confidence=0.3, note=f"дата не прочитана целиком; в штампе виден год: {', '.join(years)}", partial=True)
    if sn: fields["series_number"] = dict(value=f"{sn['series']} {sn['number']}", series=sn["series"], number=sn["number"], confidence=round(sn["score"], 3), note="вертикальная полоса (серия/номер) на странице")
    if stamps: fields["stamps_handwriting"] = dict(value=None, confidence=0.0, note="рукописные записи (адрес, даты, подпись) не распознаются")
    stamp_list = _stamp_list(passes, hits)
    kind = "registration" if strong >= 1 else ("passport_page" if sn else "unknown")
    return dict(kind=kind, cues=int(sum(cues.values())), strong_cues=int(strong), stamps=int(stamps), stamp_list=stamp_list, fields=fields, series_number=sn, lines_n={k: len(v) for k, v in passes.items()})
