import io
import shutil

import pytest
from PIL import Image

import config
from core import tagging
from core.hashing import sha256_bytes

ENGINES = ["builtin"] + (["exiftool"] if shutil.which("exiftool") else [])


def pixels(data: bytes) -> bytes:
    return Image.open(io.BytesIO(data)).convert("RGBA").tobytes()


@pytest.mark.parametrize("engine", ENGINES)
def test_tag_written_and_verified(image_bytes, engine):
    before = sha256_bytes(image_bytes)
    result = tagging.add_dnt_tag(image_bytes, "Jane Doé", "https://jane.example", engine=engine)
    assert result.verified
    assert tagging.read_data_mining(result.data) == config.DMI_PROHIBITED_AIML
    assert pixels(result.data) == pixels(image_bytes)
    assert sha256_bytes(image_bytes) == before  # input untouched
    assert result.data != image_bytes


@pytest.mark.parametrize("engine", ENGINES)
def test_retagging_does_not_duplicate(image_bytes, engine):
    once = tagging.add_dnt_tag(image_bytes, "A", None, engine=engine).data
    twice = tagging.add_dnt_tag(once, "A", None, engine="builtin").data
    assert tagging.read_data_mining(twice) == config.DMI_PROHIBITED_AIML
    assert twice.count(b"DataMining") == once.count(b"DataMining")


@pytest.mark.skipif(not shutil.which("exiftool"), reason="ExifTool not installed")
def test_builtin_tag_readable_by_exiftool(image_bytes):
    tagged = tagging.add_dnt_tag(image_bytes, "Jane", None, engine="builtin").data
    meta = tagging.read_with_exiftool(tagged)
    assert meta["DataMining"] == config.DMI_PROHIBITED_AIML or meta["DataMining"].endswith("DMI-PROHIBITED-AIMLTRAINING")
    assert meta["Creator"] == "Jane"


def test_existing_xmp_is_kept(jpeg_bytes):
    packet = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
              '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="5"/>'
              "</rdf:RDF></x:xmpmeta>")
    with_xmp = tagging._jpeg_set_xmp(jpeg_bytes, packet)
    tagged = tagging.add_dnt_tag(with_xmp, None, None, engine="builtin").data
    xmp = tagging.extract_xmp(tagged)
    assert "Rating" in xmp and "DMI-PROHIBITED-AIMLTRAINING" in xmp


def test_untagged_reads_none(image_bytes):
    assert tagging.read_data_mining(image_bytes) is None


def test_rejects_unknown_format():
    with pytest.raises(ValueError):
        tagging.add_dnt_tag(b"GIF89a....", engine="builtin")
