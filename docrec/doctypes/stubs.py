"""Заготовки будущих типов: пока умеют только опознаваться (classify). Для extract — см. README."""
from .base import DocType, register


@register
class SNILS(DocType):
    name = "snils"; title_ru = "СНИЛС"
    keywords = [("СТРАХОВОЕ СВИДЕТЕЛЬСТВО", 3), ("ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ", 2), ("СНИЛС", 1)]
