"""Translation module using Sugoi V4 for offline Japanese to English."""

import os
from difflib import SequenceMatcher
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from . import log

logger = log.get_logger()

# Suppress HuggingFace Hub warning about unauthenticated requests
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Official HuggingFace repository for Sugoi V4
SUGOI_REPO_ID = "entai2965/sugoi-v4-ja-en-ctranslate2"

# Translation cache defaults
DEFAULT_CACHE_SIZE = 10000            # Max cached translations
DEFAULT_SIMILARITY_THRESHOLD = 0.9  # Fuzzy match threshold for cache lookup


def _get_sugoi_model_path() -> Path:
    """Get path to Sugoi V4 translation model, downloading if needed.

    Downloads from official HuggingFace source on first use.
    Model is cached in standard HuggingFace cache (~/.cache/huggingface/).

    Returns:
        Path to the model directory.
    """
    # First try to load from cache (no network request)
    try:
        model_path = snapshot_download(
            repo_id=SUGOI_REPO_ID,
            local_files_only=True,
        )
        return Path(model_path)
    except LocalEntryNotFoundError:
        pass

    # Not cached, download from HuggingFace
    logger.info("downloading sugoi v4 model", size="~1.1GB")
    model_path = snapshot_download(repo_id=SUGOI_REPO_ID)
    return Path(model_path)


def text_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class TranslationCache:
    """LRU cache for translations with fuzzy key matching."""

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        cache_file: str = None
    ):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store.
            similarity_threshold: Minimum similarity ratio for fuzzy match (0.0-1.0).
        """
        self._cache: dict[str, str] = {}
        self._max_size = max_size
        self._similarity_threshold = similarity_threshold
        
        if cache_file:
            # load cache entries from the passed file
            import csv
            with open(cache_file, 'r') as csv_file:
                logger.info("loading translation cache from " + cache_file)
                csv_sniffer = csv.Sniffer().sniff(csv_file.read(1024))
                csv_file.seek(0)
                csv_file_reader = csv.reader(csv_file, dialect=csv_sniffer)
                for curr_row in csv_file_reader:
                    if len(curr_row) < 2:
                        continue
                    #logger.debug(curr_row[0])
                    #logger.debug(curr_row[1])
                    self.put(self._normalize_for_hash(curr_row[0]), ">"+curr_row[1])
                # end for
                logger.info("loaded %d entries in translation cache" % (len(self._cache)))

    def get(self, text: str) -> str | None:
        """Get cached translation, using fuzzy matching if exact match not found.

        Args:
            text: Japanese text to look up.

        Returns:
            Cached translation if found, None otherwise.
        """
        # Try exact match first
        if text in self._cache:
            return self._cache[text]
        
        # Try normalized
        logger.info("original: " + text)
        text = self._normalize_for_hash(text)
        logger.info("normalized: " + text)
        if text in self._cache:
            return self._cache[text]

        # Try fuzzy match
        prev_cached_text = ""
        for cached_text, translation in self._cache.items():
            if text_similarity(text, cached_text) >= self._similarity_threshold:
                return translation
            #if text in cached_text:  # is a substring
            #    return translation
            # combine with prev string
            if text_similarity(text, prev_cached_text+cached_text) >= self._similarity_threshold:
                #logger.info("combined match: " + prev_cached_text+cached_text)
                return translation
            prev_cached_text = cached_text

        return None

    def put(self, text: str, translation: str) -> None:
        """Store a translation in the cache.

        Args:
            text: Japanese source text.
            translation: English translation.
        """
        # Simple LRU: remove oldest entry if at capacity
        if len(self._cache) >= self._max_size and text not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[text] = translation

    def _normalize_for_hash(self, text: str) -> str :
        import re
        import unicodedata
        
        if not text:
            return ""
        
        # NFKC Normalization (Handles Full-width/Half-width and basic compositions)
        text = unicodedata.normalize('NFKC', text)
        
        # Decompose to separate Dakuten (NFD)
        text = unicodedata.normalize('NFD', text)
        
        # Remove Dakuten/Handakuten (U+3099, U+309A)
        text = re.sub(r'[\u3099\u309A]', '', text)
        
        # Small Kana to Large Kana Mapping
        small_to_large = str.maketrans(
            'ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ',
            'あいうえおつやゆよわアイウエオツヤユヨワカケ'
        )
        text = text.translate(small_to_large)
        
        # Visual Lookalike Mapping (Fuzzy Canonicalization)
        lookalike_map = str.maketrans({
            '二': 'ニ',  # Kanji Two -> Katakana Ni
            '工': 'エ',  # Kanji Work -> Katakana E
            '口': 'ロ',  # Kanji Mouth -> Katakana Ro
            '一': 'ー',  # Kanji One -> Long Vowel Mark
            '|': '1',   # Pipe -> One
            'l': '1',   # Lowercase L -> One
            'O': '0',   # Letter O -> Zero
            '入': '人',  # Enter -> Person
            '士': '土',  # Warrior -> Soil
            '末': '未',  # End -> Not yet
            '干': '千',  # Dry -> 1000
            '自': '白',  # Self -> White
            '曰': '日',  # Say -> Sun
        })
        text = text.translate(lookalike_map)
        
        # Remove all Punctuation, Spaces, and Symbols
        # Keeps: Hiragana, Katakana, Kanji, and Alpha-numeric
        #text = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF0-9a-zA-Z]', '', text)
        # FINAL CLEANING: Keep ONLY Alphanumeric, Kanji, Hiragana, and Katakana.
        # This explicitly removes '.' and '・' and any other symbols/punctuation.
        # Range breakdown:
        # 0-9a-zA-Z : Standard Alphanumeric
        # \u3040-\u309F : Hiragana
        # \u30A0-\u30FF : Katakana
        # \u4E00-\u9FFF : Kanji
        text = re.sub(r'[^0-9a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', text)
    
        return text
        
        

class Translator:
    """Translates Japanese text to English using Sugoi V4 (CTranslate2)."""

    def __init__(
        self,
        cache_size: int = DEFAULT_CACHE_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        cache_file: str = None,
    ):
        """Initialize the translator (lazy loading).

        Args:
            cache_size: Maximum number of translations to cache.
            similarity_threshold: Minimum similarity for fuzzy cache match (0.0-1.0).
        """
        self._model_path = None
        self._translator = None
        self._tokenizer = None
        self._cache = TranslationCache(cache_size, similarity_threshold, cache_file)

    def load(self) -> None:
        """Load the translation model, downloading if needed."""
        if self._translator is not None:
            return

        logger.info("loading sugoi v4")

        import ctranslate2
        import sentencepiece as spm

        # Get model path (downloads from HuggingFace if needed)
        self._model_path = _get_sugoi_model_path()

        # Load CTranslate2 model with GPU if available, fallback to CPU
        device = "cpu"
        try:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                # Try to load with GPU
                self._translator = ctranslate2.Translator(
                    str(self._model_path),
                    device="cuda",
                )
                device = "cuda"
        except Exception:
            # GPU failed, will use CPU below
            pass

        if device == "cpu":
            self._translator = ctranslate2.Translator(
                str(self._model_path),
                device="cpu",
            )

        # Load SentencePiece tokenizer
        tokenizer_path = self._model_path / "spm" / "spm.ja.nopretok.model"
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.Load(str(tokenizer_path))

        device_info = "GPU" if device == "cuda" else "CPU"
        logger.info("sugoi v4 ready", device=device_info)

        
    def translate(self, text: str) -> tuple[str, bool]:
        """Translate Japanese text to English.

        Args:
            text: Japanese text to translate.

        Returns:
            Tuple of (translated English text, was_cached).
        """
        
        if not text or not text.strip():
            return "", False
        
        # skip lines with only ascii chars
        if text.isascii():
            return "", False
                
        # Check cache first (includes fuzzy matching)
        cached = self._cache.get(text)
        if cached is not None:
            return cached, True

        # Ensure model is loaded
        if self._translator is None:
            self.load()

        # Tokenize input
        tokens = self._tokenizer.EncodeAsPieces(text)

        # Translate
        results = self._translator.translate_batch(
            [tokens],
            beam_size=5,
            max_decoding_length=256,
        )

        # Decode output - join tokens and clean up SentencePiece markers
        translated_tokens = results[0].hypotheses[0]
        result = "".join(translated_tokens).replace("▁", " ").strip()

        # Normalize Unicode characters to ASCII equivalents (fixes rendering issues)
        result = (
            result
            # Curly quotes → straight quotes
            .replace("\u2018", "'")   # LEFT SINGLE QUOTATION MARK
            .replace("\u2019", "'")   # RIGHT SINGLE QUOTATION MARK
            .replace("\u201C", '"')   # LEFT DOUBLE QUOTATION MARK
            .replace("\u201D", '"')   # RIGHT DOUBLE QUOTATION MARK
            # Dashes
            .replace("\u2013", "-")   # EN DASH
            .replace("\u2014", "--")  # EM DASH
            .replace("\u2212", "-")   # MINUS SIGN
            # Spaces
            .replace("\u00A0", " ")   # NO-BREAK SPACE
            # Ellipsis
            .replace("\u2026", "...")  # HORIZONTAL ELLIPSIS
        )

        # Store in cache
        self._cache.put(text, result)

        return result, False

    def is_loaded(self) -> bool:
        """Check if the translation model is loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._translator is not None
