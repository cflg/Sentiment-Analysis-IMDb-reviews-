import re
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from preprocessing import clear_text


def test_clear_text():
    input_text = "This <b>AMAZING</b> Movie, is full of surprises!!! And <i>THE</i> plot: ..."
    result = clear_text(input_text)

    # Lowercase
    assert result == result.lower()

    # HTML tags removed
    assert '<' not in result and '>' not in result

    # Punctuation stripped
    assert not re.search(r"[^a-z\s]", result)

    # Stop-word removal
    for stop_word in ["this", "is", "full", "and", "the"]:
        assert stop_word not in result.split()

    # Final cleaned text
    assert result == "amazing movie surprises plot"
