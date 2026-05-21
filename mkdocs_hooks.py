import posixpath
import re


IMG_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)


def _rewrite_local_img_src(src: str, page) -> str:
    if src.startswith(("http://", "https://", "/", "#", "data:")):
        return src

    source_dir = posixpath.dirname(page.file.src_uri)
    docs_relative_target = posixpath.normpath(posixpath.join(source_dir, src))

    # Only rewrite files that live under docs/assets/; keep other relative paths untouched.
    if not docs_relative_target.startswith("assets/"):
        return src

    current_page_dir = page.url.rstrip("/") or "."
    return posixpath.relpath(docs_relative_target, start=current_page_dir)


def on_page_content(html, page, config, files):
    def repl(match):
        prefix, src, suffix = match.groups()
        return f"{prefix}{_rewrite_local_img_src(src, page)}{suffix}"

    return IMG_SRC_RE.sub(repl, html)
