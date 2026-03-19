"""The implementation of chrF (Popović 2015) and chrF++ (Popović 2017) metrics."""

def sum_of_lists(lists):
    """Aggregates list of numeric lists by summing."""
    if len(lists) == 1:
        return lists[0]

    # Preserve datatype
    size = len(lists[0])
    init_val = type(lists[0][0])(0.0)
    total = [init_val] * size
    for ll in lists:
        for i in range(size):
            total[i] += ll[i]
    return total

"""The base `Score`, `Metric` and `Signature` classes to derive from.

`Metric` is an abstract class that enforces the implementation of a set
of abstract methods. This way, a correctly implemented metric will work
seamlessly with the rest of the codebase.
"""

import json
import logging
import statistics
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

sacrelogger = logging.getLogger("sacrebleu_methods")


class Score:
    """A base score class to derive from.

    :param name: The name of the underlying metric.
    :param score: A floating point number for the final metric.
    """

    def __init__(self, name: str, score: float):
        """`Score` initializer."""
        self.name = name
        self.score = score

        # Statistical test related fields
        self._mean = -1.0
        self._ci = -1.0

        # More info can be added right after the score
        self._verbose = ""

    def format(
        self,
        width: int = 2,
        score_only: bool = False,
        signature: str = "",
        is_json: bool = False,
    ) -> str:
        """Returns a pretty representation of the score.
        :param width: Floating point decimal precision width.
        :param score_only: If `True`, and the format is not `json`,
        returns a single score string.
        :param signature: A string representation of the given `Signature`
        instance.
        :param is_json: If `True`, will output the score in JSON string.
        :return: A plain or JSON-formatted string representation.
        """
        d = {
            "name": self.name,
            "score": float(f"{self.score:.{width}f}"),
            "signature": signature,
        }

        sc = f"{self.score:.{width}f}"

        if self._mean > 0:
            confidence_mean = f"{self._mean:.{width}f}"
            confidence_var = f"{self._ci:.{width}f}"
            confidence_str = f"μ = {confidence_mean} ± {confidence_var}"

            sc += f" ({confidence_str})"
            if is_json:
                d["confidence_mean"] = float(confidence_mean)
                d["confidence_var"] = float(confidence_var)
                d["confidence"] = confidence_str

        # Construct full score line
        full_score = f"{self.name}|{signature}" if signature else self.name
        full_score = f"{full_score} = {sc}"
        if self._verbose:
            full_score += f" {self._verbose}"
            d["verbose_score"] = self._verbose

        if score_only:
            return sc

        if is_json:
            for param in signature.split("|"):
                key, value = param.split(":")
                d[key] = value
            return json.dumps(d, indent=1, ensure_ascii=False)

        return full_score

    def estimate_ci(self, scores: List["Score"]):
        """Takes a list of scores and stores mean, stdev and 95% confidence
        interval around the mean.

        :param scores: A list of `Score` objects obtained from bootstrap
        resampling for example.
        """
        # Sort the scores
        raw_scores = sorted([x.score for x in scores])
        n = len(raw_scores)

        # Get CI bounds (95%, i.e. 1/40 from left)
        lower_idx = n // 40
        upper_idx = n - lower_idx - 1
        lower, upper = raw_scores[lower_idx], raw_scores[upper_idx]
        self._ci = 0.5 * (upper - lower)
        self._mean = statistics.mean(raw_scores)

    def __repr__(self):
        """Returns a human-readable score string."""
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    :param args: key-value dictionary passed from the actual metric instance.
    """

    def __init__(self, args: dict):
        """`Signature` initializer."""
        # Global items that are shared across all metrics
        self._abbr = {
            "version": "v",
            "nrefs": "#",
            "test": "t",
            "lang": "l",
            "subset": "S",
            "origlang": "o",
            "bs": "bs",  # Bootstrap resampling trials
            "ar": "ar",  # Approximate randomization trials
            "seed": "rs",  # RNG's seed
        }

        if "num_refs" not in args:
            raise RuntimeError("Number of references unknown, please evaluate the metric first.")

        num_refs = args["num_refs"]
        if num_refs == -1:
            # Detect variable number of refs
            num_refs = "var"

        # Global items that are shared across all metrics
        # None's will be ignored
        self.info = {
            # "version": __version__,
            "nrefs": num_refs,
            "bs": args.get("n_bootstrap", None),
            "ar": None,
            "seed": args.get("seed", None),
            "test": args.get("test_set", None),
            "lang": args.get("langpair", None),
            "origlang": args.get("origlang", None),
            "subset": args.get("subset", None),
        }

    def format(self, short: bool = False) -> str:
        """Returns a string representation of the signature.

        :param short: If True, shortened signature is produced.
        :return: A string representation of the signature.
        """
        pairs = []
        keys = list(self.info.keys())
        # keep version always at end
        keys.remove("version")
        for name in keys + ["version"]:
            value = self.info[name]
            if value is not None:
                if isinstance(value, bool):
                    # Replace True/False with yes/no
                    value = "yes" if value else "no"
                final_name = self._abbr[name] if short else name
                pairs.append(f"{final_name}:{value}")

        return "|".join(pairs)

    def update(self, key: str, value: Any):
        """Add a new item or update an existing one.

        :param key: The key to use in the dictionary.
        :param value: The associated value for the `key`.
        """
        self.info[key] = value

    def __str__(self):
        """Returns a human-readable signature string."""
        return self.format()

    def __repr__(self):
        """Returns a human-readable signature string."""
        return self.format()


class Metric(metaclass=ABCMeta):
    """A base class for all metrics that ensures the implementation of some
    methods. Much of the common functionality is moved to this base class
    from other metrics."""

    # Each metric should define its Signature class' name here
    _SIGNATURE_TYPE = Signature

    def __init__(self):
        """`Metric` initializer."""
        # The pre-computed reference cache
        self._ref_cache = None

        # only useful for BLEU tokenized warnings. Set to True so that
        # warnings are not issued for other metrics.
        self._force = True

        # Will be used by the signature when bootstrap resampling
        self.n_bootstrap = None
        self.seed = None

    def _check_sentence_score_args(self, hyp: str, refs: Sequence[str]):
        """Performs sanity checks on `sentence_score` method's arguments.

        :param hyp: A single hypothesis string.
        :param refs: A sequence of reference strings.
        """
        prefix = self.__class__.__name__
        err_msg = None

        if not isinstance(hyp, str):
            err_msg = "The argument `hyp` should be a string."
        elif isinstance(refs, str) or not isinstance(refs, Sequence):
            err_msg = "The argument `refs` should be a sequence of strings."
        elif not isinstance(refs[0], str):
            err_msg = "Each element of `refs` should be a string."

        if err_msg:
            raise RuntimeError(f"{prefix}: {err_msg}")

    def _check_corpus_score_args(self, hyps: Sequence[str], refs: Optional[Sequence[Sequence[str]]]):
        """Performs sanity checks on `corpus_score` method's arguments.

        :param hypses: A sequence of hypothesis strings.
        :param refs: A sequence of reference documents with document being
        defined as a sequence of reference strings. If `None`, cached references
        will be used.
        """

        prefix = self.__class__.__name__
        err_msg = None

        if not isinstance(hyps, Sequence):
            err_msg = "`hyps` should be a sequence of strings."
        elif not isinstance(hyps[0], str):
            err_msg = "Each element of `hyps` should be a string."
        elif any(line is None for line in hyps):
            err_msg = "Undefined line in hypotheses stream!"

        if refs is not None:
            # print(type(refs))
            # print(type(refs[0]))
            # print(type(refs[0][0]))
            # print(isinstance(refs[0], Sequence))
            if not isinstance(refs, Sequence):
                err_msg = "`refs` should be a sequence of sequence of strings."
            elif not isinstance(refs[0], Sequence):
                err_msg = "Each element of `refs` should be a sequence of strings."
            elif not isinstance(refs[0][0], str):
                err_msg = "`refs` should be a sequence of sequence of strings."

        if err_msg:
            raise RuntimeError(f"{prefix}: {err_msg}")

    @abstractmethod
    def _aggregate_and_compute(self, stats: List[List[Any]]) -> Any:
        """Computes the final score given the pre-computed match statistics.

        :param stats: A list of segment-level statistics.
        :return: A `Score` instance.
        """
        pass

    @abstractmethod
    def _compute_score_from_stats(self, stats: List[Any]) -> Any:
        """Computes the final score from already aggregated statistics.

        :param stats: A list or numpy array of segment-level statistics.
        :return: A `Score` object.
        """
        pass

    @abstractmethod
    def _preprocess_segment(self, sent: str) -> str:
        """A wrapper around the metric's tokenization and pre-processing logic.
        This should be implemented for reference caching to work correctly.

        :param sent: The input sentence.
        :return: The pre-processed output sentence.
        """
        pass

    @abstractmethod
    def _extract_reference_info(self, refs: Sequence[str]) -> Dict[str, Any]:
        """Given a list of reference segments, extract the required
        information (such as n-grams for BLEU and chrF). This should be implemented
        for the generic `_cache_references()` to work across all metrics.

        :param refs: A sequence of strings.
        """
        pass

    @abstractmethod
    def _compute_segment_statistics(self, hypothesis: str, ref_kwargs: Dict) -> List[Any]:
        """Given a (pre-processed) hypothesis sentence and already computed
        reference info, returns the best match statistics across the
        references. The return type is usually a List of ints or floats.

        :param hypothesis: A pre-processed hypothesis sentence.
        :param ref_kwargs: A dictionary with reference-related information
        within. This is formulated as a dictionary as different metrics may
        require different information regarding a reference segment.
        """
        pass

    def _cache_references(self, references: Sequence[Sequence[str]]) -> List[Any]:
        """Given the full set of document references, extract segment n-grams
        (or other necessary information) for caching purposes.

        :param references: A sequence of reference documents with document being
        defined as a sequence of reference strings. A particular reference
        segment can be '' or `None` to allow the use of variable number
        of references per segment.
        :return: A list where each element is a tuple of segment n-grams and
        reference lengths, as returned by `_extract_reference_info()`.
        """
        ref_cache = []

        # Decide on final number of refs here as well
        num_refs = set()

        for refs in zip(*references):
            # remove undefined / empty references
            # i.e. we have fewer references for this particular sentence
            lines = [x for x in refs if x is not None and x != ""]

            if len(lines) == 0:
                raise RuntimeError("Empty or `None` reference sentence found.")

            # Keep track of reference counts to allow variable reference
            # info in the signature
            num_refs.add(len(lines))

            lines = [self._preprocess_segment(x) for x in lines]

            # Get n-grams
            ref_cache.append(self._extract_reference_info(lines))

        if len(num_refs) == 1:
            self.num_refs = list(num_refs)[0]
        else:
            # A variable number of refs exist
            self.num_refs = -1

        return ref_cache

    def _extract_corpus_statistics(
        self, hypotheses: Sequence[str], references: Optional[Sequence[Sequence[str]]]
    ) -> Any:
        """Reads the corpus and returns sentence-level match statistics for
        faster re-computations esp. during statistical tests.

        :param hypotheses: A sequence of hypothesis strings.
        :param references: A sequence of reference documents with document being
        defined as a sequence of reference strings. If `None`, cached references
        will be used.
        :return: A list where each sublist corresponds to segment statistics.
        """
        # Pre-compute references
        # Don't store the cache as the user is explicitly passing refs
        if references:
            ref_cache = self._cache_references(references)
        elif self._ref_cache:
            ref_cache = self._ref_cache
        else:
            raise RuntimeError("No references provided and the cache is empty.")

        stats = []
        tok_count = 0

        for hyp, ref_kwargs in zip(hypotheses, ref_cache):
            # Check for already-tokenized input problem (only for BLEU)
            if not self._force and hyp.endswith(" ."):
                tok_count += 1

            hyp = self._preprocess_segment(hyp)

            # Collect stats
            stats.append(self._compute_segment_statistics(hyp, ref_kwargs))

        if tok_count >= 100:
            sacrelogger.warning("That's 100 lines that end in a tokenized period ('.')")
            sacrelogger.warning("It looks like you forgot to detokenize your test data, which may hurt your score.")
            sacrelogger.warning(
                "If you insist your data is detokenized, or don't care, you can suppress this message with the `force` parameter."
            )

        return stats

    def sentence_score(self, hypothesis: str, references: Sequence[str]) -> Any:
        """Compute the metric for a single sentence against a single (or multiple) reference(s).

        :param hypothesis: A single hypothesis string.
        :param references: A sequence of reference strings.
        :return: A `Score` object.
        """
        self._check_sentence_score_args(hypothesis, references)

        stats = self._extract_corpus_statistics([hypothesis], [[refs] for refs in references])
        return self._aggregate_and_compute(stats)

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Optional[Sequence[Sequence[str]]],
        n_bootstrap: int = 1,
    ) -> Any:
        """Compute the metric for a corpus against a single (or multiple) reference(s).

        :param hypotheses: A sequence of hypothesis strings.
        :param references: A sequence of reference documents with document being
        defined as a sequence of reference strings. If `None`, cached references
        will be used.
        :param n_bootstrap: If > 1, provides 95% confidence interval around true mean
        using bootstrap resampling with `n_bootstrap` samples.
        :return: A `Score` object.
        """
        # print(len(hypotheses))
        # print(len(references))
        # print(len(references[0]))
        self._check_corpus_score_args(hypotheses, references)

        # Collect corpus stats
        stats = self._extract_corpus_statistics(hypotheses, references)

        # Compute the actual system score
        actual_score = self._aggregate_and_compute(stats)

        if n_bootstrap > 1:
            # Compute bootstrap estimate as well
            # Delayed import is to escape from numpy import if bootstrap
            # is not requested.
            from ..significance import _bootstrap_resample

            self.n_bootstrap = n_bootstrap
            self.seed, bs_scores = _bootstrap_resample(stats, self, n_bootstrap)
            actual_score.estimate_ci(bs_scores)

        return actual_score

    def get_signature(self) -> Signature:
        """Creates and returns the signature for the metric. The creation
        of signatures is delayed as the number of references is resolved
        only at the point of reference caching."""
        return self._SIGNATURE_TYPE(self.__dict__)

"""Various utility functions for word and character n-gram extraction."""

from collections import Counter
from typing import List, Tuple


def extract_all_word_ngrams(line: str, min_order: int, max_order: int) -> Tuple[Counter, int]:
    """Extracts all ngrams (min_order <= n <= max_order) from a sentence.

    :param line: A string sentence.
    :param min_order: Minimum n-gram order.
    :param max_order: Maximum n-gram order.
    :return: a Counter object with n-grams counts and the sequence length.
    """

    ngrams = []
    tokens = line.split()

    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))

    return Counter(ngrams), len(tokens)


def extract_word_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts n-grams with order `n` from a list of tokens.

    :param tokens: A list of tokens.
    :param n: The order of n-grams.
    :return: a Counter object with n-grams counts.
    """
    return Counter([" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def extract_char_ngrams(line: str, n: int, include_whitespace: bool = False) -> Counter:
    """Yields counts of character n-grams from a sentence.

    :param line: A segment containing a sequence of words.
    :param n: The order of the n-grams.
    :param include_whitespace: If given, will not strip whitespaces from the line.
    :return: a dictionary containing ngrams and counts
    """
    if not include_whitespace:
        line = "".join(line.split())

    return Counter([line[i : i + n] for i in range(len(line) - n + 1)])


def extract_all_char_ngrams(line: str, max_order: int, include_whitespace: bool = False) -> List[Counter]:
    """Extracts all character n-grams at once for convenience.

    :param line: A segment containing a sequence of words.
    :param max_order: The maximum order of the n-grams.
    :param include_whitespace: If given, will not strip whitespaces from the line.
    :return: a list of Counter objects containing ngrams and counts.
    """

    counters = []

    if not include_whitespace:
        line = "".join(line.split())

    for n in range(1, max_order + 1):
        ngrams = Counter([line[i : i + n] for i in range(len(line) - n + 1)])
        counters.append(ngrams)

    return counters


class CHRFSignature(Signature):
    """A convenience class to represent the reproducibility signature for chrF.

    :param args: key-value dictionary passed from the actual metric instance.
    """

    def __init__(self, args: dict):
        """`CHRFSignature` initializer."""
        super().__init__(args)
        self._abbr.update(
            {
                "case": "c",
                "eff": "e",
                "nc": "nc",
                "nw": "nw",
                "space": "s",
            }
        )

        self.info.update(
            {
                "case": "lc" if args["lowercase"] else "mixed",
                "eff": "yes" if not args["eps_smoothing"] else "no",
                "nc": args["char_order"],
                "nw": args["word_order"],
                "space": "yes" if args["whitespace"] else "no",
            }
        )


class CHRFScore(Score):
    """A convenience class to represent chrF scores.

    :param score: The chrF (chrF++) score.
    :param char_order: The character n-gram order.
    :param word_order: The word n-gram order. If equals to 2, the metric is referred to as chrF++.
    :param beta: Determine the importance of recall w.r.t precision.
    """

    def __init__(self, score: float, char_order: int, word_order: int, beta: int):
        """`CHRFScore` initializer."""
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order

        # Add + signs to denote chrF+ variant
        name = f"chrF{self.beta}" + "+" * self.word_order

        super().__init__(name, score)


class CHRF(Metric):
    """Computes the chrF(++) metric given hypotheses and references.

    :param char_order: Character n-gram order.
    :param word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
    :param beta: Determine the importance of recall w.r.t precision.
    :param lowercase: Enable case-insensitivity.
    :param whitespace: If `True`, include whitespaces when extracting character n-grams.
    :param eps_smoothing: If `True`, applies epsilon smoothing similar
    to reference chrF++.py, NLTK and Moses implementations. Otherwise,
    it takes into account effective match order similar to sacreBLEU < 2.0.0.
    :param references: A sequence of reference documents with document being
    defined as a sequence of reference strings. If given, the reference n-grams
    will be pre-computed and cached for faster re-computation across many systems.
    """

    # Maximum character n-gram order to take into account
    CHAR_ORDER = 6

    # chrF+ additionally takes into account some of the word n-grams
    WORD_ORDER = 0

    # Defaults to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    # Cache string.punctuation for chrF+' punctuation stripper
    _PUNCTS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    _SIGNATURE_TYPE = CHRFSignature

    def __init__(
        self,
        char_order: int = CHAR_ORDER,
        word_order: int = WORD_ORDER,
        beta: int = BETA,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
        references: Optional[Sequence[Sequence[str]]] = None,
    ):
        """`CHRF` initializer."""
        super().__init__()

        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = self.char_order + self.word_order
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.eps_smoothing = eps_smoothing

        if references is not None:
            # Pre-compute reference ngrams
            self._ref_cache = self._cache_references(references)

    @staticmethod
    def _get_match_statistics(hyp_ngrams: Counter, ref_ngrams: Counter) -> List[int]:
        """Computes the match statistics between hypothesis and reference n-grams.

        :param hyp_ngrams: A `Counter` holding hypothesis n-grams.
        :param ref_ngrams: A `Counter` holding reference n-grams.
        :return: A list of three numbers denoting hypothesis n-gram count,
            reference n-gram count and the intersection count.
        """
        # Counter's internal intersection is not that fast, count manually
        match_count, hyp_count = 0, 0
        for ng, count in hyp_ngrams.items():
            hyp_count += count
            if ng in ref_ngrams:
                match_count += min(count, ref_ngrams[ng])

        return [
            # Don't count hits if no reference exists for that n-gram
            hyp_count if ref_ngrams else 0,
            sum(ref_ngrams.values()),
            match_count,
        ]

    def _remove_punctuation(self, sent: str) -> List[str]:
        """Separates out punctuations from beginning and end of words for chrF.
        Adapted from https://github.com/m-popovic/chrF

        :param sent: A string.
        :return: A list of words.
        """
        tokenized = []
        for w in sent.split():
            if len(w) == 1:
                tokenized.append(w)
            else:
                # NOTE: This splits '(hi)' to '(hi' and ')' (issue #124)
                if w[-1] in self._PUNCTS:
                    tokenized += [w[:-1], w[-1]]
                elif w[0] in self._PUNCTS:
                    tokenized += [w[0], w[1:]]
                else:
                    tokenized.append(w)
        return tokenized

    def _preprocess_segment(self, sent: str) -> str:
        """Given a sentence, apply optional lowercasing.

        :param sent: The input sentence string.
        :return: The pre-processed output string.
        """
        return sent.lower() if self.lowercase else sent

    def _compute_f_score(self, statistics: List[int]) -> float:
        """Compute the chrF score given the n-gram match statistics.

        :param statistics: A flattened list of 3 * (`char_order` + `word_order`)
            elements giving the [hyp, ref, match] counts for each order.
        :return: The final f_beta score between [0, 100].
        """
        eps = 1e-16
        score = 0.0
        effective_order = 0
        factor = self.beta**2
        avg_prec, avg_rec = 0.0, 0.0

        for i in range(self.order):
            n_hyp, n_ref, n_match = statistics[3 * i : 3 * i + 3]

            # chrF++.py style EPS smoothing (also used by Moses and NLTK)
            prec = n_match / n_hyp if n_hyp > 0 else eps
            rec = n_match / n_ref if n_ref > 0 else eps

            denom = factor * prec + rec
            score += ((1 + factor) * prec * rec / denom) if denom > 0 else eps

            # sacreBLEU <2.0.0 style effective order smoothing
            if n_hyp > 0 and n_ref > 0:
                avg_prec += prec
                avg_rec += rec
                effective_order += 1

        if self.eps_smoothing:
            return 100 * score / self.order

        if effective_order == 0:
            avg_prec = avg_rec = 0.0
        else:
            avg_prec /= effective_order
            avg_rec /= effective_order

        if avg_prec + avg_rec:
            score = (1 + factor) * avg_prec * avg_rec
            score /= (factor * avg_prec) + avg_rec
            return 100 * score
        else:
            return 0.0

    def _compute_score_from_stats(self, stats: List[int]) -> CHRFScore:
        """Computes the final score from already aggregated statistics.

        :param stats: A list or numpy array of segment-level statistics.
        :return: A `CHRFScore` object.
        """
        return CHRFScore(self._compute_f_score(stats), self.char_order, self.word_order, self.beta)

    def _aggregate_and_compute(self, stats: List[List[int]]) -> CHRFScore:
        """Computes the final score given the pre-computed corpus statistics.

        :param stats: A list of segment-level statistics
        :return: A `CHRFScore` object.
        """
        return self._compute_score_from_stats(sum_of_lists(stats))

    def _extract_reference_info(self, refs: Sequence[str]) -> Dict[str, List[List[Counter]]]:
        """Given a list of reference segments, extract the character and word n-grams.

        :param refs: A sequence of reference segments.
        :return: A list where each element contains n-grams per reference segment.
        """
        ngrams = []

        for ref in refs:
            # extract character n-grams
            stats = extract_all_char_ngrams(ref, self.char_order, self.whitespace)

            # Check chrF+ mode
            if self.word_order > 0:
                ref_words = self._remove_punctuation(ref)

                for n in range(self.word_order):
                    stats.append(extract_word_ngrams(ref_words, n + 1))

            ngrams.append(stats)

        return {"ref_ngrams": ngrams}

    def _compute_segment_statistics(self, hypothesis: str, ref_kwargs: Dict) -> List[int]:
        """Given a (pre-processed) hypothesis sentence and already computed
        reference n-grams, returns the best match statistics across the
        references.

        :param hypothesis: Hypothesis sentence.
        :param ref_kwargs: A dictionary with key `ref_ngrams` which is a list
        where each sublist contains n-gram counters for a particular reference sentence.
        :return: A list of integers where each triplet denotes [hyp, ref, match]
        statistics.
        """
        best_stats = []
        best_f_score = -1.0

        # extract character n-grams
        all_hyp_ngrams = extract_all_char_ngrams(hypothesis, self.char_order, self.whitespace)

        # Check chrF+ mode to see if we'll add word n-grams as well
        if self.word_order > 0:
            # Primitive tokenization: separate out punctuations
            hwords = self._remove_punctuation(hypothesis)
            _range = range(1, self.word_order + 1)
            all_hyp_ngrams.extend([extract_word_ngrams(hwords, n) for n in _range])

        # Iterate over multiple references, pick the one with best F score
        for _ref_ngrams in ref_kwargs["ref_ngrams"]:
            stats = []
            # Traverse all orders
            for h, r in zip(all_hyp_ngrams, _ref_ngrams):
                stats.extend(self._get_match_statistics(h, r))
            f_score = self._compute_f_score(stats)

            if f_score > best_f_score:
                best_f_score = f_score
                best_stats = stats

        return best_stats


def sentence_chrf(
    hypothesis: str,
    references: Sequence[str],
    char_order: int = CHRF.CHAR_ORDER,
    word_order: int = CHRF.WORD_ORDER,
    beta: int = CHRF.BETA,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    """
    Computes chrF for a single sentence against a single (or multiple) reference(s).
    If `word_order` equals to 2, the metric is referred to as chrF++.

    :param hypothesis: A single hypothesis string.
    :param references: A sequence of reference strings.
    :param char_order: Character n-gram order.
    :param word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
    :param beta: Determine the importance of recall w.r.t precision.
    :param eps_smoothing: If `True`, applies epsilon smoothing similar
    to reference chrF++.py, NLTK and Moses implementations. Otherwise,
    it takes into account effective match order similar to sacreBLEU < 2.0.0.
    :param remove_whitespace: If `True`, removes whitespaces prior to character n-gram extraction.
    :return: A `CHRFScore` object.
    """
    metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    )
    return metric.sentence_score(hypothesis, references)

