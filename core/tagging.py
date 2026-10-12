"""Write and verify "Do Not Train" XMP metadata on copies of image files.

Two engines:
- "exiftool": the ExifTool binary (preferred; merges cleanly with existing metadata).
- "builtin": pure-Python XMP packet insertion for JPEG, PNG and WebP. It never
  re-encodes pixels: it only adds or replaces the XMP segment/chunk.

Both engines write the IPTC/PLUS DataMining property with the full PLUS URI.
Callers always pass bytes and get new bytes back; originals are never touched.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

import config

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "plus": "http://ns.useplus.org/ldf/xmp/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)

JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
PNG_XMP_KEYWORD = b"XML:com.adobe.xmp"
EXIFTOOL_DMI_PRINT_VALUE = "Prohibited for AI/ML training"


@dataclass
class TagResult:
    data: bytes
    engine: str
    verified: bool
    data_mining_value: str | None


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def rights_text(artist: str | None, website: str | None) -> str:
    parts = [config.DNT_NOTICE]
    if artist:
        parts.append(f"Creator: {artist}.")
    if website:
        parts.append(f"Contact: {website}.")
    return " ".join(parts)


def detect_format(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Unsupported image format (expected JPEG, PNG or WebP).")


def add_dnt_tag(data: bytes, artist: str | None = None, website: str | None = None,
                engine: str = "auto") -> TagResult:
    """Return a tagged copy of `data`, verified by reading the tag back."""
    if engine == "auto":
        engine = "exiftool" if exiftool_available() else "builtin"
    if engine == "exiftool":
        tagged = _tag_with_exiftool(data, artist, website)
    elif engine == "builtin":
        tagged = _tag_builtin(data, artist, website)
    else:
        raise ValueError(f"Unknown tagging engine: {engine}")
    value = read_data_mining(tagged)
    return TagResult(tagged, engine, value == config.DMI_PROHIBITED_AIML, value)


# ---------------------------------------------------------------- reading

def extract_xmp(data: bytes) -> str | None:
    """Return the first XMP packet embedded in a JPEG, PNG or WebP file."""
    fmt = detect_format(data)
    if fmt == "jpeg":
        for marker, payload, _start, _end in _jpeg_segments(data):
            if marker == 0xE1 and payload.startswith(JPEG_XMP_HEADER):
                return payload[len(JPEG_XMP_HEADER):].decode("utf-8", "replace")
    elif fmt == "png":
        for ctype, body, _start, _end in _png_chunks(data):
            if ctype == b"iTXt" and body.startswith(PNG_XMP_KEYWORD + b"\x00"):
                return _parse_itxt(body)
    else:
        for fourcc, body, _start, _end in _webp_chunks(data):
            if fourcc == b"XMP ":
                return body.decode("utf-8", "replace")
    return None


def read_data_mining(data: bytes) -> str | None:
    xmp = extract_xmp(data)
    if not xmp:
        return None
    try:
        root = ET.fromstring(_strip_xpacket(xmp))
    except ET.ParseError:
        return None
    key = f"{{{NS['plus']}}}DataMining"
    for desc in root.iter(f"{{{NS['rdf']}}}Description"):
        if key in desc.attrib:
            return desc.attrib[key].strip()
        el = desc.find(key)
        if el is not None and el.text:
            return el.text.strip()
    return None


def read_with_exiftool(data: bytes) -> dict:
    """Read the tags back with ExifTool (used by tests and the README check)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"img.{detect_format(data)}"
        path.write_bytes(data)
        out = subprocess.run(
            ["exiftool", "-j", "-n", "-XMP-plus:DataMining", "-XMP-dc:Creator", str(path)],
            capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout)[0]


# ---------------------------------------------------------------- exiftool

def _tag_with_exiftool(data: bytes, artist: str | None, website: str | None) -> bytes:
    text = rights_text(artist, website)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"img.{detect_format(data)}"
        path.write_bytes(data)
        cmd = [
            "exiftool", "-q", "-overwrite_original", "-charset", "utf8",
            f"-XMP-plus:DataMining={EXIFTOOL_DMI_PRINT_VALUE}",
            f"-XMP-plus:OtherConstraints={text}",
            f"-XMP-xmpRights:UsageTerms={text}",
            f"-XMP-dc:Rights={text}",
        ]
        if artist:
            cmd.append(f"-XMP-dc:Creator={artist}")
        cmd.append(str(path))
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        return path.read_bytes()


# ---------------------------------------------------------------- builtin

def _tag_builtin(data: bytes, artist: str | None, website: str | None) -> bytes:
    fmt = detect_format(data)
    existing = extract_xmp(data)
    packet = build_xmp_packet(artist, website, existing)
    if fmt == "jpeg":
        return _jpeg_set_xmp(data, packet)
    if fmt == "png":
        return _png_set_xmp(data, packet)
    return _webp_set_xmp(data, packet)


def _description_xml(artist: str | None, website: str | None) -> str:
    text = escape(rights_text(artist, website))
    creator = ""
    if artist:
        creator = f"<dc:creator><rdf:Seq><rdf:li>{escape(artist)}</rdf:li></rdf:Seq></dc:creator>"

    def alt(tag: str) -> str:
        return f"<{tag}><rdf:Alt><rdf:li xml:lang=\"x-default\">{text}</rdf:li></rdf:Alt></{tag}>"

    return (
        f'<rdf:Description rdf:about="" xmlns:plus="{NS["plus"]}" xmlns:dc="{NS["dc"]}" '
        f'xmlns:xmpRights="{NS["xmpRights"]}">'
        f"<plus:DataMining>{config.DMI_PROHIBITED_AIML}</plus:DataMining>"
        f"{alt('plus:OtherConstraints')}{alt('xmpRights:UsageTerms')}{alt('dc:rights')}{creator}"
        "</rdf:Description>"
    )


def build_xmp_packet(artist: str | None, website: str | None, existing: str | None = None) -> str:
    """Build an XMP packet. If the file already has XMP, keep it and add our
    properties in an extra rdf:Description (dropping any old DataMining value)."""
    ours = _description_xml(artist, website)
    body = None
    if existing:
        try:
            root = ET.fromstring(_strip_xpacket(existing))
            rdf = root if root.tag == f"{{{NS['rdf']}}}RDF" else root.find(f"{{{NS['rdf']}}}RDF")
            if rdf is not None:
                _drop_our_properties(rdf)
                ns_decl = f'xmlns:rdf="{NS["rdf"]}"'
                new_desc = ET.fromstring(ours.replace("<rdf:Description ", f"<rdf:Description {ns_decl} ", 1))
                rdf.append(new_desc)
                body = ET.tostring(root if root.tag != rdf.tag else rdf, encoding="unicode")
                if root.tag == rdf.tag:
                    body = f'<x:xmpmeta xmlns:x="{NS["x"]}">{body}</x:xmpmeta>'
        except ET.ParseError:
            body = None  # unreadable existing XMP: replace it rather than corrupt it
    if body is None:
        body = (f'<x:xmpmeta xmlns:x="{NS["x"]}"><rdf:RDF xmlns:rdf="{NS["rdf"]}">'
                f"{ours}</rdf:RDF></x:xmpmeta>")
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            f"{body}<?xpacket end=\"w\"?>")


def _drop_our_properties(rdf: ET.Element) -> None:
    ours = {f"{{{NS['plus']}}}DataMining", f"{{{NS['plus']}}}OtherConstraints",
            f"{{{NS['xmpRights']}}}UsageTerms", f"{{{NS['dc']}}}rights"}
    for desc in rdf.findall(f"{{{NS['rdf']}}}Description"):
        for key in list(desc.attrib):
            if key in ours:
                del desc.attrib[key]
        for child in list(desc):
            if child.tag in ours:
                desc.remove(child)


def _strip_xpacket(xmp: str) -> str:
    xmp = re.sub(r"<\?xpacket[^>]*\?>", "", xmp)
    return xmp.strip().strip("\x00").strip()


# JPEG ----------------------------------------------------------------------

def _jpeg_segments(data: bytes):
    """Yield (marker, payload, seg_start, seg_end) for header segments up to SOS."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("Malformed JPEG")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        yield marker, data[i + 4:i + 2 + length], i, i + 2 + length
        if marker == 0xDA:  # start of scan: compressed data follows
            return
        i += 2 + length


def _jpeg_set_xmp(data: bytes, packet: str) -> bytes:
    payload = JPEG_XMP_HEADER + packet.encode("utf-8")
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("XMP packet too large for a single JPEG segment")
    xmp_segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    header: list[bytes] = []
    insert_at = 0
    scan_start = None
    for marker, seg_payload, start, end in _jpeg_segments(data):
        if marker == 0xDA:
            scan_start = start
            break
        if marker == 0xE1 and seg_payload.startswith(JPEG_XMP_HEADER):
            continue  # old XMP; replaced by ours
        header.append(data[start:end])
        if marker in (0xE0, 0xE1) and insert_at == len(header) - 1:
            insert_at = len(header)  # keep JFIF/EXIF first, as readers expect
    if scan_start is None:
        raise ValueError("Malformed JPEG: no image data found")
    header.insert(insert_at, xmp_segment)
    return data[:2] + b"".join(header) + data[scan_start:]


# PNG -----------------------------------------------------------------------

def _png_chunks(data: bytes):
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i:i + 4])
        ctype = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        yield ctype, body, i, i + 12 + length
        i += 12 + length
        if ctype == b"IEND":
            return


def _parse_itxt(body: bytes) -> str:
    keyword, rest = body.split(b"\x00", 1)
    compressed, _method = rest[0], rest[1]
    rest = rest[2:]
    _lang, rest = rest.split(b"\x00", 1)
    _translated, text = rest.split(b"\x00", 1)
    if compressed:
        text = zlib.decompress(text)
    return text.decode("utf-8", "replace")


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def _png_set_xmp(data: bytes, packet: str) -> bytes:
    body = PNG_XMP_KEYWORD + b"\x00\x00\x00\x00\x00" + packet.encode("utf-8")
    new_chunk = _png_chunk(b"iTXt", body)
    out = bytearray(data[:8])
    inserted = False
    for ctype, cbody, start, end in _png_chunks(data):
        if ctype == b"iTXt" and cbody.startswith(PNG_XMP_KEYWORD + b"\x00"):
            continue
        if ctype == b"IDAT" and not inserted:
            out += new_chunk
            inserted = True
        out += data[start:end]
    return bytes(out)


# WebP ----------------------------------------------------------------------

def _webp_chunks(data: bytes):
    i = 12
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        (size,) = struct.unpack("<I", data[i + 4:i + 8])
        end = i + 8 + size + (size & 1)
        yield fourcc, data[i + 8:i + 8 + size], i, end
        i = end


def _webp_chunk(fourcc: bytes, body: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")


def _webp_set_xmp(data: bytes, packet: str) -> bytes:
    chunks = [(f, data[s:e]) for f, _b, s, e in _webp_chunks(data) if f != b"XMP "]
    if chunks and chunks[0][0] == b"VP8X":
        vp8x = bytearray(chunks[0][1])
        vp8x[8] |= 0x04  # XMP flag (first payload byte)
        chunks[0] = (b"VP8X", bytes(vp8x))
    else:
        # Simple (lossy/lossless) WebP: add a VP8X header so extra chunks are allowed.
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            has_alpha = "A" in im.getbands()
        flags = 0x04 | (0x10 if has_alpha else 0)
        payload = bytes([flags, 0, 0, 0]) + (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little")
        chunks.insert(0, (b"VP8X", _webp_chunk(b"VP8X", payload)))
    body = b"WEBP" + b"".join(c for _f, c in chunks) + _webp_chunk(b"XMP ", packet.encode("utf-8"))
    return b"RIFF" + struct.pack("<I", len(body)) + body
