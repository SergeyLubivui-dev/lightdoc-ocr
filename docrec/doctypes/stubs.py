"""Заготовки будущих типов: пока умеют только опознаваться (classify). Для extract — см. README."""
from .base import DocType, register


@register
class PassportRF(DocType):
    name = "passport_rf"; title_ru = "Паспорт гражданина РФ (разворот с фото)"
    card_size = (1500, 1056); aspect = 125 / 88
    keywords = [("ПАСПОРТ ВЫДАН", 2), ("РОССИЙСКАЯ ФЕДЕРАЦИЯ", 1), ("КОД ПОДРАЗДЕЛЕНИЯ", 2), ("ДАТА РОЖДЕНИЯ", 1), ("МЕСТО РОЖДЕНИЯ", 1), ("ФАМИЛИЯ", 0.5)]


@register
class VehicleRegistrationSTS(DocType):
    name = "sts"; title_ru = "Свидетельство о регистрации ТС (СТС)"
    keywords = [("СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ", 3), ("ТРАНСПОРТНОГО СРЕДСТВА", 2), ("ИДЕНТИФИКАЦИОННЫЙ НОМЕР VIN", 1), ("РЕГИСТРАЦИОННЫЙ ЗНАК", 1)]


@register
class SNILS(DocType):
    name = "snils"; title_ru = "СНИЛС"
    keywords = [("СТРАХОВОЕ СВИДЕТЕЛЬСТВО", 3), ("ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ", 2), ("СНИЛС", 1)]
