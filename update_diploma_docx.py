# -*- coding: utf-8 -*-
import copy
import random
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}

ET.register_namespace("", W_NS)
ET.register_namespace("w14", W14_NS)


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def new_hex8() -> str:
    return f"{random.randrange(16**8):08X}"


def para_text(p: ET.Element) -> str:
    parts = []
    for t in p.findall(".//w:t", NS):
        if t.text:
            parts.append(t.text)
    return "".join(parts).strip()


def clone_rpr(template: ET.Element) -> Optional[ET.Element]:
    run = template.find("w:r", NS)
    if run is None:
        return None
    rpr = run.find("w:rPr", NS)
    if rpr is None:
        return None
    return copy.deepcopy(rpr)


def strip_para_children(p: ET.Element) -> None:
    ppr = p.find("w:pPr", NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)


def make_run(text: str, rpr: Optional[ET.Element]) -> ET.Element:
    run = ET.Element(qn(W_NS, "r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    t = ET.SubElement(run, qn(W_NS, "t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t.set(qn(XML_NS, "space"), "preserve")
    t.text = text
    return run


def make_page_break_run(rpr: Optional[ET.Element]) -> ET.Element:
    run = ET.Element(qn(W_NS, "r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    br = ET.SubElement(run, qn(W_NS, "br"))
    br.set(qn(W_NS, "type"), "page")
    return run


def build_paragraph(template: ET.Element, text: str, page_break: bool = False) -> ET.Element:
    p = copy.deepcopy(template)
    p.set(qn(W14_NS, "paraId"), new_hex8())
    p.set(qn(W14_NS, "textId"), new_hex8())
    strip_para_children(p)
    rpr = clone_rpr(template)
    if page_break:
        p.append(make_page_break_run(rpr))
    p.append(make_run(text, rpr))
    return p


def replace_paragraph(body: ET.Element, old: ET.Element, new: ET.Element) -> None:
    children = list(body)
    idx = children.index(old)
    body.remove(old)
    body.insert(idx, new)


def insert_before(body: ET.Element, anchor: ET.Element, new_items: list[ET.Element]) -> None:
    idx = list(body).index(anchor)
    for offset, item in enumerate(new_items):
        body.insert(idx + offset, item)


def find_paragraph(body: ET.Element, text: str) -> ET.Element:
    for child in list(body):
        if child.tag == qn(W_NS, "p") and para_text(child) == text:
            return child
    raise ValueError(f"Paragraph not found: {text}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: py update_diploma_docx.py <input.docx> <output.docx>")
        return 1

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    with zipfile.ZipFile(src, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    root = ET.fromstring(files["word/document.xml"])
    body = root.find(".//w:body", NS)
    if body is None:
        raise RuntimeError("Document body not found")

    chapter3_heading_old = find_paragraph(body, "3.  ТЕСТИРОВАНИЕ")
    conclusion_heading_old = find_paragraph(body, "ЗАКЛЮЧЕНИЕ")
    literature_heading_old = find_paragraph(body, "СПИСОК ИСПОЛЬЗОВАННОЙ ЛИТЕРАТУРЫ")

    chapter_heading_template = chapter3_heading_old
    section_heading_template = find_paragraph(body, "2.1. Общая архитектура системы")
    body_template = find_paragraph(
        body,
        "Система реализована как модульный пайплайн: каждый этап обработки вынесен в отдельный Python-модуль со строго определённым интерфейсом. Такое решение продиктовано практической необходимостью — в ходе разработки отдельные компоненты приходилось переписывать и тестировать независимо, не трогая остальные. Монолитный скрипт в таких условиях быстро превращается в неуправляемый код.",
    )

    chapter3_heading_new = build_paragraph(
        chapter_heading_template,
        "3. ТЕСТИРОВАНИЕ И ОЦЕНКА КАЧЕСТВА СИСТЕМЫ",
        page_break=True,
    )
    conclusion_heading_new = build_paragraph(
        chapter_heading_template,
        "ЗАКЛЮЧЕНИЕ",
        page_break=True,
    )
    literature_heading_new = build_paragraph(
        chapter_heading_template,
        "СПИСОК ЛИТЕРАТУРЫ",
        page_break=True,
    )

    replace_paragraph(body, chapter3_heading_old, chapter3_heading_new)
    replace_paragraph(body, conclusion_heading_old, conclusion_heading_new)
    replace_paragraph(body, literature_heading_old, literature_heading_new)

    chapter3_blocks: list[tuple[str, str]] = [
        ("h", "3.1. Постановка эксперимента"),
        (
            "p",
            "Экспериментальная проверка разработанной системы выполнялась на сохранённых артефактах проекта и результатах запусков тестовых скриптов, размещённых в каталоге data/test. В качестве исходного набора для сравнения использовался файл segments_man.json, содержащий 115 речевых фрагментов, полученных после этапа автоматического распознавания речи и сегментации.",
        ),
        (
            "p",
            "Для части экспериментов применялась стратегия sentence-level, при которой соседние фрагменты предварительно объединялись в полные предложения. В этом случае число единиц сравнения сокращалось до 103, что важно учитывать при сопоставлении результатов с per-segment режимом.",
        ),
        (
            "p",
            "Основной метрикой оценки выступало среднее косинусное сходство LaBSE между исходным английским текстом и русским переводом. Выбор этой метрики обусловлен отсутствием заранее подготовленных эталонных переводов и необходимостью оценивать именно семантическую близость, а не буквальное совпадение формулировок.",
        ),
        (
            "p",
            "Численные результаты сравнения стратегий перевода были зафиксированы в файле translation_comparison_man.json. Результаты сравнения моделей и режимов перевода были получены из файлов models_comparison_man.json, models_strategies_comparison_man.json и comparison_with_google_man.json. Дополнительно для анализа few-shot fine-tuning XTTS-v2 использовались сохранённые результаты исследовательского ноутбука final.ipynb.",
        ),
        (
            "p",
            "Следует отметить, что в модуле src/metrics.py также реализованы вычисление WER, CER и speaker verification score, однако в доступных артефактах репозитория численные значения этих метрик не были сохранены в отдельной итоговой сводке. Поэтому в данной редакции главы основной акцент сделан на сравнении переводческих стратегий и моделей, а сведения о дообучении XTTS-v2 приведены как дополнительный эксперимент.",
        ),
        ("h", "3.2. Сравнение стратегий перевода"),
        (
            "p",
            "На первом этапе тестирования было выполнено сравнение трёх стратегий машинного перевода: независимого перевода каждого сегмента, скользящего окна и sentence-level перевода, при котором несколько соседних сегментов предварительно объединяются в законченное предложение.",
        ),
        (
            "p",
            "Результаты показали, что стратегия per-segment обеспечила среднее значение LaBSE 0.8609 при 115 сегментах, strategy sentence-level — 0.8590 при 103 сегментах, а sliding-window — лишь 0.5446 при тех же 115 сегментах. Минимальные и максимальные значения для per-segment составили 0.5503 и 0.9510, для sentence-level — 0.5503 и 0.9510, для sliding-window — 0.0222 и 0.9150.",
        ),
        (
            "p",
            "Столь сильное падение качества у sliding-window объясняется не только качеством собственно перевода, сколько трудностями обратного разбиения общего переведённого блока на исходные сегменты. При нарушении границ фраз отдельные сегменты теряют смысловую целостность, что немедленно отражается на семантической метрике.",
        ),
        (
            "p",
            "Стратегии per-segment и sentence-level продемонстрировали практически одинаковый результат. Однако sentence-level режим имеет важное практическое преимущество: в модуль синтеза поступают не фрагменты незавершённых фраз, а полноценные предложения, благодаря чему облегчается последующая временная синхронизация и снижается риск неестественного звучания коротких синтетических реплик.",
        ),
        ("h", "3.3. Сравнение моделей перевода"),
        (
            "p",
            "На втором этапе было проведено сравнение двух дистиллированных версий NLLB-200: модели на 600 млн параметров и модели на 1.3 млрд параметров. Для обеих моделей оценивались два режима работы: per-segment и sentence-level.",
        ),
        (
            "p",
            "Конфигурация NLLB-600M × per-segment показала среднее значение LaBSE 0.8609 при времени перевода около 12.0 с, а NLLB-600M × sentence-level — 0.8590 при времени 10.4 с. Для более крупной модели NLLB-1.3B результаты составили 0.8732 и 350.5 с в режиме per-segment, а также 0.8689 и 331.1 с в режиме sentence-level.",
        ),
        (
            "p",
            "Лучший результат по качеству показала конфигурация NLLB-1.3B × per-segment, для которой среднее значение LaBSE составило 0.8732. Однако выигрыш относительно NLLB-600M × per-segment оказался умеренным и составил около 0.0123 по абсолютной шкале. Одновременно время перевода выросло многократно: примерно с 12 до 350 секунд в сохранённом запуске расширенного сравнения.",
        ),
        (
            "p",
            "Полученные данные позволяют сделать два вывода. Во-первых, увеличение размера модели действительно улучшает перевод, но прирост качества не является радикальным. Во-вторых, в условиях ограниченных вычислительных ресурсов более компактная модель NLLB-600M может рассматриваться как практический компромисс между качеством и скоростью.",
        ),
        ("h", "3.4. Сопоставление с внешним базовым решением"),
        (
            "p",
            "Для дополнительной оценки полученные результаты были сопоставлены с переводом через Google Translate, который использовался как внешний базовый ориентир. Среднее значение LaBSE для Google Translate составило 0.8715 при минимуме 0.5850 и максимуме 0.9597.",
        ),
        (
            "p",
            "Таким образом, Google Translate оказался сильнее, чем конфигурации на основе NLLB-600M, но незначительно уступил лучшему локальному варианту NLLB-1.3B × per-segment с результатом 0.8732. Это подтверждает, что выбранная исследовательская постановка является корректной: локальный пайплайн следует сравнивать не только с внутренними альтернативами, но и с промышленными сервисами общего назначения.",
        ),
        (
            "p",
            "В то же время именно модель NLLB-1.3B сохранила первое место по качеству. Следовательно, её использование в составе автономного локального пайплайна оправдано, если задачей является максимальная независимость от внешних API при сохранении качества перевода, сопоставимого с сильным внешним baseline.",
        ),
        ("h", "3.5. Дополнительный эксперимент по few-shot fine-tuning XTTS-v2"),
        (
            "p",
            "Поскольку во второй главе рассматривалась возможность дообучения XTTS-v2 на голосе конкретного диктора, в работе был зафиксирован вспомогательный эксперимент, результаты которого сохранены в ноутбуке final.ipynb. В этом эксперименте сравнивались zero-shot и few-shot генерации на основе косинусного сходства внутренних представлений модели.",
        ),
        (
            "p",
            "Для цельно сгенерированных аудиодорожек длительностью 15 секунд среднее значение speaker embedding составило 0.6395, а среднее значение GPT conditioning latent — 0.7717. Для десяти случайных сегментов текста выступления длительностью 6-12 секунд эти показатели составили 0.7033 и 0.7824 соответственно.",
        ),
        (
            "p",
            "Полученные значения показывают, что после few-shot дообучения более устойчиво сохраняется близость по GPT conditioning latent, чем по speaker embedding. Иными словами, дообученная модель в большей степени удерживает интонационно-просодическую структуру, тогда как тембральные характеристики меняются заметнее. Этот результат нельзя трактовать как прямую оценку сходства с оригинальным диктором, однако он подтверждает, что few-shot адаптация действительно изменяет внутреннее акустическое представление голоса.",
        ),
        ("h", "3.6. Выводы по результатам тестирования"),
        (
            "p",
            "Проведённое тестирование подтвердило работоспособность предложенного пайплайна и позволило выявить наиболее удачные конфигурации его ключевых компонентов. Стратегия sliding-window оказалась непригодной для практического использования из-за существенного падения семантической точности. Подходы per-segment и sentence-level показали сопоставимое качество, однако sentence-level режим оказался более удобным для последующего синтеза и синхронизации.",
        ),
        (
            "p",
            "Наилучшее качество перевода в сохранённых артефактах обеспечила модель NLLB-1.3B в режиме per-segment. Вместе с тем компактная модель NLLB-600M продемонстрировала близкий уровень качества при значительно меньших вычислительных затратах, что позволяет рекомендовать выбор конфигурации в зависимости от доступных аппаратных ресурсов и требований к скорости работы.",
        ),
        (
            "p",
            "Дополнительное сравнение с Google Translate показало, что локальный пайплайн на основе NLLB-1.3B способен достигать качества, сопоставимого с сильным внешним сервисом машинного перевода. Вспомогательный эксперимент по few-shot fine-tuning XTTS-v2 подтвердил, что адаптация модели меняет акустическое представление речи и может использоваться как перспективное направление дальнейшего развития проекта.",
        ),
    ]

    conclusion_blocks: list[tuple[str, str]] = [
        (
            "p",
            "Выполненная работа была посвящена разработке системы автоматического дубляжа видео с английского языка на русский с сохранением голосовых характеристик исходного спикера. В ходе исследования были рассмотрены теоретические основы автоматического распознавания речи, машинного перевода, синтеза речи, клонирования голоса и разделения аудиоисточников. На их основе был спроектирован и реализован модульный программный пайплайн, включающий предобработку аудио, сегментацию речи, перевод, синтез озвучки, постобработку и сборку итогового видео.",
        ),
        (
            "p",
            "Практическая часть работы показала, что предложенный подход является реализуемым и даёт количественно измеримые результаты. Сравнение стратегий перевода подтвердило неэффективность sliding-window подхода для рассматриваемой постановки, тогда как per-segment и sentence-level режимы обеспечили близкое качество. Лучший зафиксированный результат по метрике LaBSE был получен для конфигурации NLLB-1.3B × per-segment и составил 0.8732. Одновременно было показано, что модель NLLB-600M может использоваться как вычислительно более доступный компромисс при умеренном снижении качества.",
        ),
        (
            "p",
            "Сопоставление с Google Translate показало, что автономная локальная конфигурация на основе NLLB-1.3B обеспечивает качество перевода, близкое к сильному внешнему baseline. Дополнительный эксперимент по few-shot fine-tuning XTTS-v2 подтвердил, что адаптация модели изменяет внутреннее акустическое представление речи и может служить перспективным инструментом улучшения персонализации синтеза.",
        ),
        (
            "p",
            "Практическая значимость работы заключается в создании функционирующего прототипа системы, пригодного для локализации образовательных и публичных видеоматериалов. Разработанный пайплайн можно использовать как основу для дальнейших исследований и инженерного развития.",
        ),
        (
            "p",
            "К числу направлений дальнейшей работы целесообразно отнести полное сохранение и повторный расчёт метрик WER, CER и speaker verification score в едином экспериментальном контуре, приведение production-конфигурации перевода в соответствие с лучшими зафиксированными результатами, автоматизацию сборки окружения проекта, а также расширение системы за счёт поддержки многоспикерных сценариев, улучшенной временной синхронизации и полноценной интеграции модуля субтитров.",
        ),
    ]

    chapter3_paragraphs = [
        build_paragraph(section_heading_template, text) if kind == "h" else build_paragraph(body_template, text)
        for kind, text in chapter3_blocks
    ]
    conclusion_paragraphs = [
        build_paragraph(body_template, text)
        for _, text in conclusion_blocks
    ]

    insert_before(body, conclusion_heading_new, chapter3_paragraphs)
    insert_before(body, literature_heading_new, conclusion_paragraphs)

    files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
