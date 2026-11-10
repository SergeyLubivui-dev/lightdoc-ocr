"""Базовый класс типа документа и реестр. Добавить новый тип = создать подкласс и @register."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
from ..ocr import Line
from ..textutils import partial_ratio

REGISTRY: dict[str, "DocType"] = {}


def register(cls):
    REGISTRY[cls.name] = cls()
    return cls


@dataclass
class Field:
    value: object
    confidence: float = 0.0
    note: str | None = None
    valid: bool | None = None
    partial: bool | None = None

    def as_dict(self):
        d = {"value": self.value, "confidence": round(float(self.confidence), 3)}
        if self.valid is not None: d["valid"] = self.valid
        if self.partial: d["partial"] = True
        if self.note: d["note"] = self.note
        return d


@dataclass
class Extraction:
    fields: dict[str, Field] = field(default_factory=dict)
    photo_zone: tuple[float, float, float, float] | None = None  # относительные координаты на нормализованной карточке
    extra: dict = field(default_factory=dict)


class DocType:
    name: ClassVar[str] = "base"
    title_ru: ClassVar[str] = ""
    card_size: ClassVar[tuple[int, int]] = (1026, 648)  # px нормализованной карточки (12 px/мм для ID-1)
    aspect: ClassVar[float] = 1026 / 648
    # (фраза, вес): нечёткий поиск по строкам OCR
    keywords: ClassVar[list[tuple[str, float]]] = []
    implemented: ClassVar[bool] = False
    expected_fields: ClassVar[list[str]] = []
    hybrid_internal: ClassVar[bool] = False   # экстрактор сам решает, когда звать запасной движок

    def classify(self, lines: list[Line]) -> float:
        """0..1 — уверенность, что это данный тип. Базовая версия: взвешенные ключевые слова."""
        if not self.keywords:
            return 0.0
        tot = sum(w for _, w in self.keywords); got = 0.0
        for phrase, w in self.keywords:
            r = max((partial_ratio(phrase, l.text) for l in lines), default=0.0)
            if r >= 0.78:
                got += w * min(1.0, (r - 0.78) / 0.17)
        return min(1.0, got / tot)

    def extract(self, card: np.ndarray, lines: list[Line], rec=None, cell=None, ocr=None, paddle=None) -> Extraction:  # pragma: no cover
        ex = Extraction()
        ex.fields["raw_lines"] = Field([l.text for l in lines], 0.0, "extractor for this type is not implemented yet")
        return ex


def in_zone(l: Line, z) -> bool:
    return z[0] <= l.cx <= z[2] and z[1] <= l.cy <= z[3]


def mean_conf(lines: list[Line]) -> float:
    return float(np.mean([l.score for l in lines])) if lines else 0.0
