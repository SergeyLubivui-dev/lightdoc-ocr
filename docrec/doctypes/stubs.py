"""Заготовки будущих типов: пока умеют только опознаваться (classify). Для extract — см. README."""
from .base import DocType, register


@register
class SNILS(DocType):
    name = "snils"; title_ru = "СНИЛС"
    keywords = [("СТРАХОВОЕ СВИДЕТЕЛЬСТВО", 3), ("ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ", 2), ("СНИЛС", 1), ("ПЕНСИОННОГО СТРАХОВАНИЯ", 1), ("ФАМИЛИЯ ИМЯ ОТЧЕСТВО", 0.5), ("ДАТА И МЕСТО РОЖДЕНИЯ", 0.5)]
    # извлечение полей пока не реализовано; лексиконы — docrec/lexicons/snils/


@register
class TractorLicense(DocType):
    """Удостоверение тракториста-машиниста (Гостехнадзор). Пока только опознаётся; лексиконы — docrec/lexicons/tractor_license/."""
    name = "tractor_license"; title_ru = "Удостоверение тракториста-машиниста (тракторист)"
    keywords = [("УДОСТОВЕРЕНИЕ ТРАКТОРИСТА", 3), ("ТРАКТОРИСТА-МАШИНИСТА", 3), ("ГОСТЕХНАДЗОР", 2), ("САМОХОДНЫХ МАШИН", 2), ("ИНСПЕКЦИЯ ГОСТЕХНАДЗОРА", 1)]


@register
class InternationalPermit(DocType):
    """Международное водительское удостоверение (ru/en/fr). Пока только опознаётся; лексиконы — docrec/lexicons/international_permit/."""
    name = "international_permit"; title_ru = "Международное водительское удостоверение"
    keywords = [("МЕЖДУНАРОДНОЕ ВОДИТЕЛЬСКОЕ", 3), ("INTERNATIONAL DRIVING PERMIT", 3), ("PERMIS DE CONDUIRE INTERNATIONAL", 3), ("CONVENTION ON ROAD TRAFFIC", 1)]
