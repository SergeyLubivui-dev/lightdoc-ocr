# lightdoc-ocr

**RU** · [English below](#english)

Лёгкая офлайн-система распознавания российских документов. Только CPU, без облака: картинки и текст никуда не отправляются.

## Что умеет

- **Водительское удостоверение РФ** (карточка 2011+, лицевая сторона): поиск рамки и выпрямление перспективы, определение типа документа,
  поля (ФИО кириллицей и латиницей, даты, место рождения, орган выдачи, номер, регион, категории по ячейкам), вырезание фото водителя
  с проверкой лица.
- **Произвольная страница / договор** (`free_text`): автоповорот 90/180/270°, порядок чтения с учётом колонок, фильтр шума от печатей,
  исправление «двойников» кириллица↔латиница, реквизиты по каждой стороне договора с проверкой контрольных сумм:
  ИНН (10/12), КПП, БИК, р/с и к/с (ключ по БИК), СНИЛС, ОГРН/ОГРНИП, паспорт, телефоны, даты, e-mail, организации, ФИО.
- Результат — JSON (у каждого поля `confidence`, у реквизитов `valid`) + картинки (`card.jpg`, `photo.jpg`, `debug*.jpg`) + `text.txt`.
- Заготовки (только опознаются): паспорт РФ, СТС, СНИЛС-карта.

## Стек

Python 3.11+, **Tesseract 5.5** через `tesserocr` (tessdata_best, rus+eng; движок внутри wheel), запасной движок
**RapidOCR PP-OCRv5 mobile** (onnxruntime), OpenCV (поиск карточки, перспектива), **YuNet** (проверка лица). Все модели лежат в репозитории
(`docrec/models`, ~60 МБ), интернет после установки не нужен.

## Требования к железу

Только CPU, GPU не нужен; 2 ядер достаточно (Tesseract однопоточный). Пик RAM: **~150–185 МБ** (Tesseract), до **~350 МБ**, если сработал
запасной проход Paddle. Замер на Ryzen 9 5900X (Windows, 1 процесс): ВУ ≈ **1.5 с**, страница A4 (1600×1200) ≈ **2.1 с**.
`--fast` (tessdata_fast) быстрее примерно на 30 %, но чуть хуже по кириллице.

## Установка

```bash
# Linux/macOS                      # Windows (PowerShell)
sh setup.sh                        .\setup.ps1
```
или вручную: `python -m venv .venv && pip install -r requirements.txt`. `rapidocr` необязателен (без него работает только Tesseract).

## Использование

```bash
python recognize.py photo.jpg --out out/                  # --mode auto (по умолчанию)
python recognize.py contract.jpg --out out/ --mode page   # лист / договор: текст + реквизиты
python recognize.py licence.jpg --out out/ --mode card    # удостоверение
# --engine tess|paddle   --fast   --threads N   --quiet
```
```python
from docrec import DocumentPipeline
res = DocumentPipeline(mode="auto").process("photo.jpg", "out/")   # dict = result.json
```
Windows: `run_docrec.bat <image> [out_dir]`. Тестовые изображения с вымышленными данными: `samples/` (генераторы: `tools/make_synthetic*.py`).

## Формат результата

Карточка (`examples/synth1/result.json`, вымышленные данные):
```json
{"doc_type": "driver_license_rf", "confidence": 0.95,
 "corners": {"order": "TL,TR,BR,BL", "points": [[304.1, 459.1], "..."]},
 "fields": {
   "surname": {"value": "ПЕТРОВ", "confidence": 0.99},
   "given_names": {"value": "АЛЕКСАНДР СЕРГЕЕВИЧ", "confidence": 0.99},
   "birth_date": {"value": "14.03.1987", "confidence": 1.0, "valid": true},
   "number": {"value": "16 12 345678", "confidence": 0.96, "valid": true},
   "categories": {"value": ["B"], "confidence": 0.85, "valid": true}},
 "photo": {"face_detected": true}, "timings_sec": {"total": 1.1}, "peak_rss_mb": 183}
```
Страница (`examples/synth_page/result.json`): `text`, `lines[]`, `parties[]` (`role`, `name`, `requisites[]`), плоский `requisites[]`:
```json
{"type": "inn", "value": "7708123450", "valid": true, "confidence": 0.947, "party": "Заказчик"}
{"type": "settlement_account", "value": "40702810700001234567", "valid": true, "confidence": 0.947, "party": "Заказчик"}
```
Также: `page_rotation_applied_ccw`, `validation_issues`, `noise_filtered`, `warnings`, `engine_fallback`.
Если контрольная сумма не сходится, а исправление одной цифры из типичных путаниц OCR единственно, значение исправляется и в `note` пишется причина.

## Как добавить тип документа

1. Создайте `docrec/doctypes/my_doc.py`:
```python
from .base import DocType, Extraction, Field, register

@register
class MyDoc(DocType):
    name = "my_doc"
    card_size = (1026, 648)                      # размер нормализованной карточки, px
    keywords = [("КЛЮЧЕВАЯ ФРАЗА", 3), ("ЕЩЁ ФРАЗА", 1)]   # (фраза, вес) для классификатора
    expected_fields = ["number"]                 # для оценки «слабого» результата

    def extract(self, card, lines, rec=None, cell=None) -> Extraction:
        ex = Extraction()
        ex.fields["number"] = Field("123", 0.9, valid=True)   # зоны задаются долями карточки
        return ex
```
2. Добавьте импорт в `docrec/doctypes/__init__.py`. Классификатор, выпрямление и вызов `extract` подхватятся из реестра.
   `lines` — строки OCR с координатами (`x0,y0,x1,y1,cx,cy,score`), `cell(img, whitelist, psm)` — перечитать маленький кроп.
3. Отладка: `debug_card.jpg` (зоны шаблона и боксы OCR).

## Ограничения

- ВУ 2011+, лицевая сторона; старые ВУ, оборот, ВУ с другой версткой — не проверялись.
- Tesseract плохо читает совсем мелкую латиницу (<~10 px) и сильный блик; при слабом результате включается Paddle-запас (если установлен).
- Страница: печать поверх текста закрывает слова; наклон >10–15° не выпрямляется (поворот только кратный 90°).
- Низкое разрешение (карточка < ~700 px по ширине) резко ухудшает цифры (`warnings: low_resolution`).
- Нет MRZ; паспорт, СТС, СНИЛС-карта пока только опознаются. Проверено на небольшом числе реальных и синтетических снимков — не заменяет ручной контроль.
- Не используйте для чужих документов без законного основания; обработка персональных данных регулируется законом.

## Лицензии

Код — MIT (`LICENSE`). Сторонние компоненты: Tesseract и tessdata_best (Apache-2.0), PaddleOCR / RapidOCR и модели PP-OCRv5 (Apache-2.0),
OpenCV и YuNet (Apache-2.0), onnxruntime (MIT), tesserocr (MIT). Примеры в `samples/spec3.png` — публичный образец документа (public domain / Wikimedia Commons);
лицо в `samples/face.png` — тестовое изображение из scikit-image (NASA, public domain). Все данные в `samples/` и `examples/` вымышлены.

---

<a id="english"></a>
# lightdoc-ocr (English)

Lightweight, fully offline recognition of Russian documents. CPU-only; no image or text ever leaves the machine.

## Features

- **Russian driver licence** (2011+ card, front side): card detection and perspective crop, document-type classification, fields
  (name in Cyrillic and Latin, dates, place of birth, issuer, number, region, categories read cell by cell), driver photo crop with face check.
- **Free-text pages / contracts** (`free_text`): auto-rotation (90/180/270°), column-aware reading order, stamp-noise filtering,
  Latin↔Cyrillic homoglyph fixes, per-party requisites with checksum validation: INN (10/12), KPP, BIK, settlement/correspondent accounts
  (BIK-keyed), SNILS, OGRN/OGRNIP, passport, phones, dates, e-mail, organisations, person names.
- Output: JSON (per-field `confidence`, requisites carry `valid`) + images (`card.jpg`, `photo.jpg`, `debug*.jpg`) + `text.txt`.
- Stubs (classification only): Russian passport, vehicle registration (STS), SNILS card.

## Stack

Python 3.11+, **Tesseract 5.5** via `tesserocr` (tessdata_best, rus+eng; the engine is bundled in the wheel), fallback **RapidOCR PP-OCRv5 mobile**
(onnxruntime), OpenCV (card detection, perspective), **YuNet** (face check). All models are in the repo (`docrec/models`, ~60 MB); no internet needed after install.

## Hardware

CPU only, no GPU; 2 cores are enough (Tesseract is single-threaded). Peak RAM **~150–185 MB** (Tesseract), up to **~350 MB** when the Paddle fallback kicks in.
Measured on a Ryzen 9 5900X (Windows, one process): driver licence ≈ **1.5 s**, A4 page (1600×1200) ≈ **2.1 s**. `--fast` (tessdata_fast) is ~30 % faster, slightly less accurate on Cyrillic.

## Install

```bash
sh setup.sh          # Linux/macOS          .\setup.ps1   # Windows
```
or `python -m venv .venv && pip install -r requirements.txt`. `rapidocr` is optional (Tesseract alone works without it).

## Usage

```bash
python recognize.py photo.jpg --out out/                  # --mode auto (default)
python recognize.py contract.jpg --out out/ --mode page   # page / contract: text + requisites
python recognize.py licence.jpg --out out/ --mode card
# --engine tess|paddle   --fast   --threads N   --quiet
```
```python
from docrec import DocumentPipeline
res = DocumentPipeline(mode="auto").process("photo.jpg", "out/")   # dict, same as result.json
```
Windows: `run_docrec.bat <image> [out_dir]`. Test images with fictional data are in `samples/` (generators: `tools/make_synthetic*.py`).

## Output

See `examples/synth1/result.json` (licence card, fictional data) and `examples/synth_page/result.json` (page). Card: `doc_type`, `confidence`, `corners`,
`fields{value, confidence, valid?, note?}`, `photo`, `timings_sec`, `peak_rss_mb`. Page: `text`, `lines[]`, `parties[]`, flat `requisites[]`
(`type`, `value`, `valid`, `confidence`, `party`, `note`), `page_rotation_applied_ccw`, `validation_issues`, `noise_filtered`, `engine_fallback`.
If a checksum fails and exactly one single-digit fix (from typical OCR confusions) satisfies it, the value is corrected and `note` says so.

## Adding a document type

Create `docrec/doctypes/my_doc.py` with a `@register`-ed `DocType` subclass (`name`, `card_size`, `keywords`, optional `expected_fields`, and
`extract(card, lines, rec=None, cell=None) -> Extraction`), then import it in `docrec/doctypes/__init__.py`. Classification, rectification and dispatch are picked
up from the registry (see the example in the Russian section). Debug with `debug_card.jpg`.

## Limitations

- Driver licence 2011+ front side only; older licences, the back side and other layouts are untested.
- Tesseract struggles with very small Latin text (<~10 px) and strong glare; the Paddle fallback (if installed) covers part of these cases.
- Pages: stamps over text hide words; skew >10–15° is not corrected (only 90° steps).
- Low resolution (card narrower than ~700 px) hurts digits (`warnings: low_resolution`).
- No MRZ; passport/STS/SNILS card are classification stubs. Tested on a small set of real and synthetic images — not a substitute for manual review.
- Do not process other people's documents without a legal basis; personal data processing is regulated by law.

## Licenses

Code: MIT (`LICENSE`). Third-party: Tesseract and tessdata_best (Apache-2.0), PaddleOCR / RapidOCR and PP-OCRv5 models (Apache-2.0),
OpenCV and YuNet (Apache-2.0), onnxruntime (MIT), tesserocr (MIT). `samples/spec3.png` is a public specimen (public domain / Wikimedia Commons); the face in
`samples/face.png` is a scikit-image test image (NASA, public domain). All data in `samples/` and `examples/` is fictional.
