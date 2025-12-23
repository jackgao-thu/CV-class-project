from pathlib import Path

# seg_pdf.py
# 依赖: pip install PyMuPDF
import fitz  # PyMuPDF

def compute_content_bbox(page: fitz.Page) -> fitz.Rect | None:
    rects = []

    # 文本块
    try:
        for b in page.get_text("blocks"):
            r = fitz.Rect(b[:4])
            # 过滤极小或空块
            if r.width > 0.5 and r.height > 0.5:
                rects.append(r)
    except Exception:
        pass

    # 位图图片
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            r = page.get_image_bbox(xref)
            if r and r.width > 0.5 and r.height > 0.5:
                rects.append(r)
    except Exception:
        pass

    # 矢量绘制（线、路径等）
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r and r.width > 0.5 and r.height > 0.5:
                rects.append(r)
    except Exception:
        pass

    if not rects:
        return None

    # 合并所有矩形
    bbox = rects[0]
    for r in rects[1:]:
        bbox = bbox | r
    return bbox

def crop_pdf_whitespace(src_pdf: Path, dst_pdf: Path, x_margin_mm: float = 2.0, y_margin_mm: float = 2.0):
    doc = fitz.open(src_pdf)
    x_margin_pt = x_margin_mm * 2.83465  # mm -> pt
    y_margin_pt = y_margin_mm * 2.83465  # mm -> pt

    for page in doc:
        page_rect = page.rect
        content_bbox = compute_content_bbox(page)
        if not content_bbox:
            # 无法检测到内容，跳过该页
            continue
        print(f"内容边界: {content_bbox}")
        # 加上边距，并限制在页面范围内
        cropped = fitz.Rect(
            max(page_rect.x0, content_bbox.x0 + x_margin_pt),
            max(page_rect.y0, content_bbox.y0 + y_margin_pt),
            min(page_rect.x1, content_bbox.x1 - x_margin_pt),
            min(page_rect.y1, content_bbox.y1 - y_margin_pt),
        )
        print(f"原始页尺寸: {page_rect}, 裁剪后尺寸: {cropped}")
        # 避免异常的极小裁剪框
        min_side = 5.0
        if cropped.width < min_side or cropped.height < min_side:
            continue

        # 设置裁剪框（也可同步设置 trim/art/bleed box）
        page.set_cropbox(cropped)
        try:
            page.set_trimbox(cropped)
            page.set_bleedbox(cropped)
            page.set_artbox(cropped)
        except Exception:
            pass

    # 全量保存为新文件
    doc.save(dst_pdf, deflate=True)
    doc.close()

if __name__ == "__main__":
    src = Path("/Users/zhaoyue/Documents/Courses/25秋/计算机视觉/2025秋计算机视觉课程设计/CV-class-project/课程设计Latex模版/fig/Presentation1.pdf")
    dst = src.with_name(src.stem + "_cropped.pdf")
    crop_pdf_whitespace(src, dst, x_margin_mm=20.0, y_margin_mm=10.0)
    print(f"已输出: {dst}")