"""논문 docx(document.xml) 편집 도우미: 원문 서식(본문·소제목·표 캡션·표·각주·그림)을 복제해 새 블록을 만든다.
HL=True 이면 수정·추가한 글자에 노란 형광을 칠한다(수정 대조본)."""
import html
import re

HL = True


def esc(t):
    return html.escape(t, quote=False)


def split_blocks(xml):
    i = xml.find("<w:body>") + len("<w:body>")
    j = xml.rfind("<w:sectPr")
    head, body, tail = xml[:i], xml[i:j], xml[j:]
    blocks = re.findall(r"<w:tbl>.*?</w:tbl>|<w:p[ >].*?</w:p>|<w:p/>", body, re.S)
    assert "".join(blocks) == body.replace("\n", "") or True
    return head, blocks, tail


def _rpr(extra=""):
    hl = '<w:highlight w:val="yellow"/>' if HL else ""
    return ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
            f'{extra}<w:color w:val="000000"/><w:kern w:val="0"/><w:sz w:val="22"/><w:szCs w:val="22"/>{hl}</w:rPr>')


def runs(text, base_extra="", size=None):
    """**굵게** 표기를 지원하는 런 목록."""
    out = []
    for k, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        extra = base_extra + ("<w:b/><w:bCs/>" if k % 2 else "")
        rp = _rpr(extra)
        if size:
            rp = rp.replace('<w:sz w:val="22"/><w:szCs w:val="22"/>', f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        out.append(f'<w:r>{rp}<w:t xml:space="preserve">{esc(part)}</w:t></w:r>')
    return "".join(out)


BODY_PPR = ('<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/><w:snapToGrid w:val="0"/>'
            '<w:spacing w:after="0" w:line="480" w:lineRule="auto"/><w:textAlignment w:val="baseline"/></w:pPr>')
HEAD_PPR = ('<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/><w:snapToGrid w:val="0"/>'
            '<w:spacing w:after="0" w:line="480" w:lineRule="auto"/><w:jc w:val="left"/><w:textAlignment w:val="baseline"/></w:pPr>')


def body(text):
    return f"<w:p>{BODY_PPR}{runs(text)}</w:p>"


def heading(text):
    return f'<w:p>{HEAD_PPR}{runs(text, "<w:b/><w:bCs/>", 24)}</w:p>'


def empty():
    return "<w:p/>"


def _seq(kind, num, rpr):
    return (f'<w:r>{rpr}<w:fldChar w:fldCharType="begin"/></w:r><w:r>{rpr}<w:instrText xml:space="preserve"> SEQ {kind} \\* ARABIC </w:instrText></w:r>'
            f'<w:r>{rpr}<w:fldChar w:fldCharType="separate"/></w:r><w:r>{rpr}<w:t>{num}</w:t></w:r><w:r>{rpr}<w:fldChar w:fldCharType="end"/></w:r>')


def caption_table(num, text):
    hl = '<w:highlight w:val="yellow"/>' if HL else ""
    rb = f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:sz w:val="20"/>{hl}</w:rPr>'
    rn = f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="20"/>{hl}</w:rPr>'
    return (f'<w:p><w:pPr><w:pStyle w:val="MDPI41tablecaption"/><w:keepNext/><w:ind w:left="0"/><w:jc w:val="left"/><w:rPr><w:sz w:val="20"/></w:rPr></w:pPr>'
            f'<w:r>{rb}<w:t xml:space="preserve">Table </w:t></w:r>{_seq("Table", num, rb)}<w:r>{rb}<w:t>.</w:t></w:r>'
            f'<w:r>{rn}<w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')


def caption_fig(num, text):
    hl = '<w:highlight w:val="yellow"/>' if HL else ""
    rb = f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>{hl}</w:rPr>'
    rn = (f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b w:val="0"/><w:bCs w:val="0"/>'
          f'<w:kern w:val="0"/><w:sz w:val="20"/><w:szCs w:val="20"/>{hl}</w:rPr>')
    return (f'<w:p><w:pPr><w:pStyle w:val="a6"/><w:spacing w:before="120" w:after="240" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
            f'<w:r>{rb}<w:t xml:space="preserve">Fig. </w:t></w:r>{_seq("Figure", num, rb)}<w:r>{rb}<w:t>.</w:t></w:r>'
            f'<w:r>{rn}<w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p>')


def set_seq_number(block, num):
    """기존 캡션의 SEQ 필드 캐시 번호 변경."""
    return re.sub(r'(<w:fldChar w:fldCharType="separate"/></w:r><w:r>(?:<w:rPr>.*?</w:rPr>)?<w:t>)\d+(</w:t>)', lambda m: m.group(1) + str(num) + m.group(2), block, count=1, flags=re.S)


def footer(text):
    hl = '<w:highlight w:val="yellow"/>' if HL else ""
    return ('<w:p><w:pPr><w:pStyle w:val="MDPI43tablefooter"/><w:spacing w:before="60"/><w:ind w:left="0"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:kern w:val="0"/>'
            f'<w:sz w:val="18"/><w:szCs w:val="18"/>{hl}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def table(widths, header, rows, align=None):
    """MDPI 형식 표: 위·아래 굵은 선, 머리행 아래 가는 선."""
    assert abs(sum(widths) - 9026) < 5, sum(widths)
    align = align or ["center"] * len(widths)
    hl = '<w:highlight w:val="yellow"/>' if HL else ""

    def cell(text, w, bold, top, bottom, jc):
        b = ""
        if top or bottom:
            b = "<w:tcBorders>" + (f'<w:top w:val="single" w:sz="{top}" w:space="0" w:color="auto"/>' if top else "") + \
                (f'<w:bottom w:val="single" w:sz="{bottom}" w:space="0" w:color="auto"/>' if bottom else "") + "</w:tcBorders>"
        rp = (f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="맑은 고딕" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
              f'{"<w:b/><w:bCs/>" if bold else ""}<w:kern w:val="0"/><w:sz w:val="18"/><w:szCs w:val="18"/>{hl}</w:rPr>')
        paras = "".join(f'<w:p><w:pPr><w:pStyle w:val="MDPI42tablebody"/><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>'
                        f'<w:jc w:val="{jc}"/></w:pPr><w:r>{rp}<w:t xml:space="preserve">{esc(line)}</w:t></w:r></w:p>'
                        for line in str(text).split("\n"))
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{b}<w:vAlign w:val="center"/></w:tcPr>{paras}</w:tc>'

    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    x = ('<w:tbl><w:tblPr><w:tblW w:w="9026" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
         '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
         '</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="57" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
         f'<w:tblGrid>{grid}</w:tblGrid>')
    x += '<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + "".join(
        cell(h, w, True, 8, 4, "center") for h, w in zip(header, widths)) + "</w:tr>"
    for k, r in enumerate(rows):
        last = k == len(rows) - 1
        x += '<w:tr><w:trPr><w:cantSplit/></w:trPr>' + "".join(
            cell(v, w, False, 0, 8 if last else 0, a) for v, w, a in zip(r, widths, align)) + "</w:tr>"
    return x + "</w:tbl>"


def figure(rid, cx, cy, pic_id, name):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{pic_id}" name="Figure {pic_id}"/><wp:cNvGraphicFramePr>'
            '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{pic_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>')


def replace_text(block, old, new):
    """블록 안 한 구절 교체(같은 런 안에 있어야 함). 바뀐 부분만 형광 런으로 쪼갠다."""
    m = re.search(r"(<w:r>(<w:rPr>.*?</w:rPr>)?<w:t[^>]*>)([^<]*?)" + re.escape(esc(old)) + r"([^<]*)(</w:t></w:r>)", block, re.S)
    assert m, old[:40]
    open_, rpr, pre, post, close = m.group(1), m.group(2) or "", m.group(3), m.group(4), m.group(5)
    hrpr = rpr.replace("</w:rPr>", '<w:highlight w:val="yellow"/></w:rPr>') if (HL and rpr) else rpr
    seg = (f'<w:r>{rpr}<w:t xml:space="preserve">{pre}</w:t></w:r>' if pre else "") + \
          f'<w:r>{hrpr}<w:t xml:space="preserve">{esc(new)}</w:t></w:r>' + \
          (f'<w:r>{rpr}<w:t xml:space="preserve">{post}</w:t></w:r>' if post else "")
    return block[:m.start()] + seg + block[m.end():]


def set_text(block, text):
    """본문 단락 전체를 새 글로 교체(첫 런 서식 유지)."""
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", block, re.S)
    return f'<w:p>{ppr.group(0) if ppr else BODY_PPR}{runs(text)}</w:p>'


def append_text(block, text):
    """단락 끝에 새 런을 덧붙인다(기존 서식·수식 런 유지)."""
    i = block.rfind("</w:p>")
    return block[:i] + runs(text) + block[i:]
