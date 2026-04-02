from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

from PIL import Image


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"

EMU_PER_CM = 360000
MAX_IMAGE_WIDTH_EMU = 15 * EMU_PER_CM


@dataclass
class Block:
    kind: str
    text: str = ""
    image_path: Path | None = None


def clean_text(text: str) -> str:
    value = text.replace("\r", "")
    value = value.replace("**", "")
    value = value.replace("`", "")
    if value.startswith("*") and value.endswith("*") and len(value) > 1:
        value = value[1:-1]
    return value.strip()


def parse_markdown(markdown_path: Path) -> List[Block]:
    blocks: List[Block] = []
    base_dir = markdown_path.parent

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            blocks.append(Block(kind="blank"))
            continue

        image_match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if image_match:
            blocks.append(
                Block(kind="image", image_path=(base_dir / image_match.group(1)).resolve())
            )
            continue

        if line.startswith("# "):
            blocks.append(Block(kind="title", text=clean_text(line[2:])))
            continue

        if line.startswith("УДК "):
            blocks.append(Block(kind="udk", text=clean_text(line)))
            continue

        if line.startswith("**[Ф.") or line.startswith("**[Ф"):
            blocks.append(Block(kind="authors", text=clean_text(line)))
            continue

        if line == "[Название образовательной организации]":
            blocks.append(Block(kind="org", text=line))
            continue

        if line.startswith("**Аннотация.**"):
            blocks.append(
                Block(
                    kind="abstract",
                    text=clean_text(line.replace("**Аннотация.**", "", 1)),
                )
            )
            continue

        if line.startswith("**Ключевые слова:**"):
            blocks.append(
                Block(
                    kind="keywords",
                    text=clean_text(line.replace("**Ключевые слова:**", "", 1)),
                )
            )
            continue

        if line == "Список литературы":
            blocks.append(Block(kind="refs_title", text=line))
            continue

        if line.startswith("*Рис."):
            blocks.append(Block(kind="caption", text=clean_text(line)))
            continue

        if re.match(r"^\[\d+\]", line):
            blocks.append(Block(kind="reference", text=clean_text(line)))
            continue

        blocks.append(Block(kind="paragraph", text=clean_text(line)))

    return blocks


def xml_tag(tag: str, attrs: dict[str, str] | None = None, inner: str = "") -> str:
    attr_text = ""
    if attrs:
        attr_text = "".join(f' {key}="{escape(value)}"' for key, value in attrs.items())
    return f"<{tag}{attr_text}>{inner}</{tag}>"


def paragraph_properties(
    *,
    align: str | None = None,
    first_line_twips: int | None = None,
    space_after_twips: int = 120,
    line_twips: int = 240,
) -> str:
    parts = [
        f'<w:spacing w:after="{space_after_twips}" w:line="{line_twips}" w:lineRule="auto"/>'
    ]
    if align:
        parts.append(f'<w:jc w:val="{align}"/>')
    if first_line_twips is not None:
        parts.append(f'<w:ind w:firstLine="{first_line_twips}"/>')
    return f"<w:pPr>{''.join(parts)}</w:pPr>"


def run_properties(*, bold: bool = False, italic: bool = False, size_half_points: int = 28) -> str:
    parts = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>',
        f'<w:sz w:val="{size_half_points}"/>',
        f'<w:szCs w:val="{size_half_points}"/>',
        '<w:lang w:val="ru-RU"/>',
    ]
    if bold:
        parts.append("<w:b/>")
    if italic:
        parts.append("<w:i/>")
    return f"<w:rPr>{''.join(parts)}</w:rPr>"


def text_run(text: str, *, bold: bool = False, italic: bool = False, size_half_points: int = 28) -> str:
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return (
        "<w:r>"
        f"{run_properties(bold=bold, italic=italic, size_half_points=size_half_points)}"
        f"<w:t{preserve}>{escape(text)}</w:t>"
        "</w:r>"
    )


def empty_paragraph() -> str:
    return "<w:p/>"


def simple_paragraph(
    text: str,
    *,
    align: str | None,
    first_line_twips: int | None,
    size_half_points: int = 28,
    bold: bool = False,
    italic: bool = False,
    space_after_twips: int = 120,
) -> str:
    return (
        "<w:p>"
        f"{paragraph_properties(align=align, first_line_twips=first_line_twips, space_after_twips=space_after_twips)}"
        f"{text_run(text, bold=bold, italic=italic, size_half_points=size_half_points)}"
        "</w:p>"
    )


def label_value_paragraph(
    label: str,
    value: str,
    *,
    size_half_points: int,
    align: str | None,
    first_line_twips: int | None,
) -> str:
    return (
        "<w:p>"
        f"{paragraph_properties(align=align, first_line_twips=first_line_twips)}"
        f"{text_run(label, bold=True, size_half_points=size_half_points)}"
        f"{text_run(' ' + value, size_half_points=size_half_points)}"
        "</w:p>"
    )


def build_image_paragraph(image_path: Path, rel_id: str, docpr_id: int) -> str:
    with Image.open(image_path) as image:
        width_px, height_px = image.size

    cx = MAX_IMAGE_WIDTH_EMU
    cy = int(cx * height_px / width_px)

    drawing = f"""
    <w:r>
      {run_properties()}
      <w:drawing>
        <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">
          <wp:extent cx="{cx}" cy="{cy}"/>
          <wp:effectExtent l="0" t="0" r="0" b="0"/>
          <wp:docPr id="{docpr_id}" name="Picture {docpr_id}"/>
          <wp:cNvGraphicFramePr>
            <a:graphicFrameLocks noChangeAspect="1"/>
          </wp:cNvGraphicFramePr>
          <a:graphic>
            <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:pic>
                <pic:nvPicPr>
                  <pic:cNvPr id="{docpr_id}" name="{escape(image_path.name)}"/>
                  <pic:cNvPicPr/>
                </pic:nvPicPr>
                <pic:blipFill>
                  <a:blip r:embed="{rel_id}" xmlns:r="{R_NS}"/>
                  <a:stretch><a:fillRect/></a:stretch>
                </pic:blipFill>
                <pic:spPr>
                  <a:xfrm>
                    <a:off x="0" y="0"/>
                    <a:ext cx="{cx}" cy="{cy}"/>
                  </a:xfrm>
                  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                </pic:spPr>
              </pic:pic>
            </a:graphicData>
          </a:graphic>
        </wp:inline>
      </w:drawing>
    </w:r>
    """.strip()

    return (
        "<w:p>"
        f"{paragraph_properties(align='center', first_line_twips=0, space_after_twips=60)}"
        f"{drawing}"
        "</w:p>"
    )


def build_document_xml(blocks: List[Block], image_rels: list[tuple[str, Path]]) -> str:
    body_parts: List[str] = []
    first_line = 709
    image_index = 0

    for block in blocks:
        if block.kind == "blank":
            body_parts.append(empty_paragraph())
        elif block.kind == "udk":
            body_parts.append(simple_paragraph(block.text, align="left", first_line_twips=0))
        elif block.kind == "title":
            body_parts.append(
                simple_paragraph(
                    block.text,
                    align="center",
                    first_line_twips=0,
                    bold=True,
                )
            )
        elif block.kind == "authors":
            body_parts.append(
                simple_paragraph(
                    block.text,
                    align="center",
                    first_line_twips=0,
                    bold=True,
                )
            )
        elif block.kind == "org":
            body_parts.append(simple_paragraph(block.text, align="center", first_line_twips=0))
        elif block.kind == "abstract":
            body_parts.append(
                label_value_paragraph(
                    "Аннотация.",
                    block.text,
                    size_half_points=24,
                    align="both",
                    first_line_twips=0,
                )
            )
        elif block.kind == "keywords":
            body_parts.append(
                label_value_paragraph(
                    "Ключевые слова:",
                    block.text,
                    size_half_points=24,
                    align="both",
                    first_line_twips=0,
                )
            )
        elif block.kind == "refs_title":
            body_parts.append(simple_paragraph(block.text, align="left", first_line_twips=0))
        elif block.kind == "caption":
            body_parts.append(
                simple_paragraph(
                    block.text,
                    align="center",
                    first_line_twips=0,
                    size_half_points=24,
                    italic=True,
                )
            )
        elif block.kind == "reference":
            body_parts.append(
                simple_paragraph(
                    block.text,
                    align="both",
                    first_line_twips=0,
                )
            )
        elif block.kind == "image":
            rel_id, _ = image_rels[image_index]
            image_index += 1
            body_parts.append(build_image_paragraph(block.image_path, rel_id, image_index))
        else:
            body_parts.append(
                simple_paragraph(
                    block.text,
                    align="both",
                    first_line_twips=first_line,
                )
            )

    body_parts.append(
        """
        <w:sectPr>
          <w:pgSz w:w="11906" w:h="16838"/>
          <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>
          <w:cols w:space="708"/>
          <w:docGrid w:linePitch="360"/>
        </w:sectPr>
        """.strip()
    )

    body = "".join(body_parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:body>{body}</w:body>
</w:document>
"""


def build_document_relationships(image_rels: list[tuple[str, Path]]) -> str:
    rels = [
        (
            "rId1",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles",
            "styles.xml",
        )
    ]
    rels.extend(
        (
            rel_id,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            f"media/{image_path.name}",
        )
        for rel_id, image_path in image_rels
    )
    items = "".join(
        f'<Relationship Id="{rel_id}" Type="{rel_type}" Target="{target}"/>'
        for rel_id, rel_type, target in rels
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {items}
</Relationships>
"""


def build_styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="28"/>
        <w:szCs w:val="28"/>
        <w:lang w:val="ru-RU"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120" w:line="240" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
</w:styles>
"""


def build_root_relationships() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_content_types(image_paths: list[Path]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    seen_exts = set()
    for image_path in image_paths:
        ext = image_path.suffix.lower().lstrip(".")
        if ext not in seen_exts:
            seen_exts.add(ext)
            content_type = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
            defaults.append(f'<Default Extension="{ext}" ContentType="{content_type}"/>')

    overrides = [
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    inner = "".join(defaults + overrides)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  {inner}
</Types>
"""


def build_core_xml() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Разработка системы автоматического дубляжа видео</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
"""


def build_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""


def write_docx(markdown_path: Path, docx_path: Path) -> None:
    blocks = parse_markdown(markdown_path)
    image_paths = [block.image_path for block in blocks if block.kind == "image" and block.image_path]
    image_rels = [(f"rId{index + 2}", path) for index, path in enumerate(image_paths)]

    document_xml = build_document_xml(blocks, image_rels)
    content_types = build_content_types(image_paths)
    root_rels = build_root_relationships()
    document_rels = build_document_relationships(image_rels)
    styles_xml = build_styles_xml()
    core_xml = build_core_xml()
    app_xml = build_app_xml()

    docx_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/styles.xml", styles_xml)

        for image_path in image_paths:
            archive.write(image_path, arcname=f"word/media/{image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--docx", required=True)
    args = parser.parse_args()

    write_docx(Path(args.markdown).resolve(), Path(args.docx).resolve())


if __name__ == "__main__":
    main()
