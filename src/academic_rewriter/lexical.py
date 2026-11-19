"""Learn conservative, single-word formalization from paired sentences."""

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


_TOKEN = re.compile(
    r"(?:https?://|www\.)[^\s]+|[\w.+-]+@[\w.-]+\.\w+|`+[^`]*`+|"
    r"\w+(?:[.'’/\-‐‑]\w+)*|[^\s]",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]+")
_FUNCTION_WORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have "
    "i if in is it its may might must nor not of on or our shall should so than that the "
    "their them these they this those to was we were what when where which who will with "
    "would you your".split()
)


@dataclass(frozen=True)
class LexicalEdit:
    """One replacement with offsets into the original input."""

    start: int
    end: int
    source: str
    target: str


def _replace_case(original: str, replacement: str) -> str | None:
    if original.islower():
        return replacement
    if original.istitle():
        return replacement.capitalize()
    return None


def _example(row: Mapping):
    source, target = row.get("source"), row.get("target")
    if not isinstance(source, str) or not isinstance(target, str) or not source.strip():
        raise ValueError("Each lexical example needs nonempty source and target strings")

    tokens = list(_TOKEN.finditer(source))
    if source == target:
        focus = row.get("focus")
        start = row.get("focus_start")
        if not isinstance(focus, str) or not _WORD.fullmatch(focus):
            raise ValueError("An identity example needs a single-word focus")
        if start is not None and (type(start) is not int or start < 0):
            raise ValueError("focus_start must be a nonnegative character offset")
        matches = [
            index for index, token in enumerate(tokens)
            if token.group().lower() == focus.lower() and (start is None or token.start() == start)
        ]
        if len(matches) != 1:
            raise ValueError("Identity focus must occur once or be disambiguated with focus_start")
        return tokens, matches[0], focus.lower()

    target_tokens = list(_TOKEN.finditer(target))
    differences = [
        index for index, (before, after) in enumerate(zip(tokens, target_tokens))
        if before.group() != after.group()
    ]
    if len(tokens) != len(target_tokens) or len(differences) != 1:
        raise ValueError("A positive lexical example must replace exactly one word")
    index = differences[0]
    before = tokens[index]
    original, replacement = before.group(), target_tokens[index].group()
    if (not _WORD.fullmatch(original) or not _WORD.fullmatch(replacement)
            or original.lower() == replacement.lower()
            or _replace_case(original, replacement.lower()) != replacement
            or source[:before.start()] + replacement + source[before.end():] != target):
        raise ValueError("A lexical replacement must preserve casing, punctuation and all other text")
    return tokens, index, replacement.lower()


def _context_groups(tokens, index):
    def neighboring_words(step):
        words = []
        for distance in (1, 2):
            position = index + step * distance
            if not 0 <= position < len(tokens):
                break
            word = tokens[position].group()
            if not _WORD.fullmatch(word):
                break
            words.append(word.lower())
        return words

    left, right = neighboring_words(-1), neighboring_words(1)
    adjacent = []
    if left and right and any(word not in _FUNCTION_WORDS for word in (left[0], right[0])):
        adjacent.append(("around", left[0], right[0]))
    pairs, singles = [], []
    for direction, words in (("left", left), ("right", right)):
        if len(words) == 2 and any(word not in _FUNCTION_WORDS for word in words):
            pairs.append((direction, *words))
        if words and words[0] not in _FUNCTION_WORDS:
            singles.append((direction, words[0]))
    return (adjacent, 1), (pairs, 1), (singles, 2)


class LexicalFormalizer:
    """A supervised classifier backed by local-context outcome counts."""

    def __init__(self, votes, *, training_sha256: str | None = None):
        self._votes = votes
        self.training_sha256 = training_sha256

    @classmethod
    def fit(cls, rows: Iterable[Mapping], *, training_sha256: str | None = None):
        votes = defaultdict(lambda: defaultdict(Counter))
        seen = {}
        has_positive = False
        for row in rows:
            tokens, index, outcome = _example(row)
            words = tuple(token.group().lower() for token in tokens)
            key = words, index
            if key in seen:
                if seen[key] != outcome:
                    raise ValueError("Conflicting lexical labels for the same source and focus")
                continue
            seen[key] = outcome
            word = words[index]
            for contexts, _ in _context_groups(tokens, index):
                has_positive |= bool(contexts) and word != outcome
                for context in contexts:
                    votes[word][context][outcome] += 1
        if not has_positive:
            raise ValueError("Training requires at least one word replacement with a usable context")
        return cls(votes, training_sha256=training_sha256)

    def save(self, path: str | Path):
        payload = {
            "version": 1,
            "training_sha256": self.training_sha256,
            "votes": {
                word: [[context, outcomes] for context, outcomes in sorted(contexts.items())]
                for word, contexts in self._votes.items()
            },
        }
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported lexical model version")
        try:
            votes = {
                word: {tuple(context): dict(outcomes) for context, outcomes in contexts}
                for word, contexts in payload["votes"].items()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid lexical model context counts") from exc
        if not votes or any(
            not outcomes or any(type(count) is not int or count < 1 for count in outcomes.values())
            for contexts in votes.values() for outcomes in contexts.values()
        ):
            raise ValueError("Invalid lexical model context counts")
        return cls(votes, training_sha256=payload.get("training_sha256"))

    def _predict(self, tokens, index, word):
        votes = self._votes.get(word)
        if not votes:
            return word
        for contexts, minimum in _context_groups(tokens, index):
            matches = [votes[key] for key in contexts if key in votes]
            if not matches:
                continue
            outcomes = {outcome for match in matches for outcome in match}
            if len(outcomes) != 1:
                return word
            outcome = outcomes.pop()
            if any(match[outcome] >= minimum for match in matches):
                return outcome
        return word

    def rewrite(self, text: str) -> tuple[str, list[LexicalEdit]]:
        tokens = list(_TOKEN.finditer(text))
        edits = []
        for index, token in enumerate(tokens):
            source = token.group()
            if not _WORD.fullmatch(source):
                continue
            word = source.lower()
            outcome = self._predict(tokens, index, word)
            if outcome == word or not _WORD.fullmatch(outcome) or outcome != outcome.lower():
                continue
            replacement = _replace_case(source, outcome)
            if replacement is not None:
                edits.append(LexicalEdit(token.start(), token.end(), source, replacement))

        parts = []
        offset = 0
        for edit in edits:
            parts.extend((text[offset:edit.start], edit.target))
            offset = edit.end
        return "".join(parts) + text[offset:], edits


def _decode_rows(content: bytes, path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object")
            _example(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no lexical examples")
    return rows


def read_lexical_pairs(path: str | Path) -> list[dict]:
    path = Path(path)
    return _decode_rows(path.read_bytes(), path)


def evaluate_lexical(model: LexicalFormalizer, rows: Iterable[Mapping]) -> dict:
    counts = Counter()
    for row in rows:
        tokens, index, expected = _example(row)
        focus = tokens[index]
        rewritten, edits = model.rewrite(row["source"])
        predicted = next((edit.target.lower() for edit in edits if edit.start == focus.start()),
                         focus.group().lower())
        positive = row["source"] != row["target"]
        counts["examples"] += 1
        counts["positive_examples"] += positive
        counts["exact"] += rewritten == row["target"]
        counts["correct_replacements"] += positive and predicted == expected
        counts["no_edit_examples"] += not positive
        counts["unnecessary_edits"] += not positive and bool(edits)
    if not counts["examples"]:
        raise ValueError("Evaluation requires at least one lexical example")
    return {
        **counts,
        "exact_sentence_accuracy": counts["exact"] / counts["examples"],
        "focused_replacement_accuracy": (
            counts["correct_replacements"] / counts["positive_examples"]
            if counts["positive_examples"] else None
        ),
        "unnecessary_edit_rate": (
            counts["unnecessary_edits"] / counts["no_edit_examples"]
            if counts["no_edit_examples"] else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit the classifier from training-only JSONL pairs")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate", help="Score separate, nonsynthetic lexical examples")
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    try:
        content = args.data.read_bytes()
        data_hash = hashlib.sha256(content).hexdigest()
        rows = _decode_rows(content, args.data)
        if args.command == "train":
            if any(str(row.get("split", "train")).strip().lower() not in {"train", "training"}
                   for row in rows):
                raise ValueError("Training requires examples from the training split")
            model = LexicalFormalizer.fit(rows, training_sha256=data_hash)
            model.save(args.output)
            result = {"training_examples": len(rows), "training_sha256": data_hash,
                      "model": str(args.output)}
        else:
            model = LexicalFormalizer.load(args.model)
            if data_hash == model.training_sha256 or any(
                row.get("synthetic") or row.get("training_only")
                or str(row.get("split", "")).strip().lower() in {"train", "training"}
                for row in rows
            ):
                raise ValueError("Evaluation requires separate, nonsynthetic data")
            result = evaluate_lexical(model, rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
