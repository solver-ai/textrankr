from typing import List
from typing import Dict
from typing import Callable
from typing import Union

import math

from networkx import Graph
from networkx import pagerank_scipy as pagerank

from .sentence import Sentence

from .utils import parse_text_into_sentences, parse_candidates_to_sentences
from .utils import build_sentence_graph
from .utils import sort_values


SINGLE_GRAPH_SIZE_LIMIT = 750
BATCH_GRAPH_SIZE = 500


class TextRank:
    """
        Args:
            tokenizer: a function or a functor of Callable[[str], List[str]] type.
            tolerance: a threshold for omitting edge weights.

        Example:
            tokenizer: YourTokenizer = YourTokenizer()
            textrank: TextRank = TextRank(tokenzier)
            summaries: str = textrank.summarize(your_text_here)
            print(summaries)
    """

    def __init__(
        self, tokenizer: Callable[[str], List[str]], tolerance: float = 0.05
    ) -> None:
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        self.tolerance: float = tolerance

    def summarize(
        self, text: Union[str, List], num_sentences: int = 3, verbose: bool = True
    ):
        """
            Summarizes the given text, using the textrank algorithm.

            Args:
                text: a raw text to be summarized.
                num_sentences: number of sentences in the summarization results.
                verbose: if True, it will return a summarized raw text, otherwise it will return a list of sentence texts.
        """

        # parse text
        if isinstance(text, str):
            sentences: List[Sentence] = parse_text_into_sentences(text, self.tokenizer)
        elif isinstance(text, list):
            sentences: List[Sentence] = parse_candidates_to_sentences(
                text, self.tokenizer
            )

        # build graph
        graph: Graph = build_sentence_graph(sentences, tolerance=self.tolerance)

        # run pagerank
        pageranks: Dict[Sentence, float] = pagerank(graph, weight="weight")

        # get top-k sentences
        sentences = sorted(pageranks, key=pageranks.get, reverse=True)
        sentences = sentences[:num_sentences]
        sentences = sorted(sentences, key=lambda sentence: sentence.index)

        # return summaries
        summaries = [sentence.text for sentence in sentences]
        if verbose:
            return "\n".join(summaries)
        else:
            return summaries

    def rank(
        self,
        text: Union[str, List],
        num_sentences: int = None,
        sort: bool = True,
        verbose: bool = True,
    ):
        """
            Rank sentences in given text, using the textrank algorithm.

            Args:
                text: a raw text to be summarized.
                num_sentences: number of sentences in the summarization results.
                verbose: if True, it will return a summarized raw text, otherwise it will return a list of sentence texts.
        """

        # parse text
        if isinstance(text, str):
            sentences: List[Sentence] = parse_text_into_sentences(text, self.tokenizer)
        elif isinstance(text, list):
            sentences: List[Sentence] = parse_candidates_to_sentences(
                text, self.tokenizer
            )

        if not num_sentences:
            num_sentences = len(sentences)

        def _batch_pagerank(sentences: list, tolerance: float):
            # build graph
            graph: Graph = build_sentence_graph(sentences, tolerance=tolerance)

            # run pagerank
            pageranks: Dict[Sentence, float] = pagerank(graph, weight="weight")

            # get scores
            sentences = [
                {"sentence": k.text, "index": k.index, "score": v}
                for i, (k, v) in enumerate(pageranks.items())
            ]
            scores = list(pageranks.values())

            return sentences, scores

        if num_sentences <= SINGLE_GRAPH_SIZE_LIMIT:
            sentences, scores = _batch_pagerank(
                sentences=sentences, tolerance=self.tolerance
            )
            ranks = sort_values(scores)
        else:
            batched_sentences = list()
            batched_scores = list()
            num_batches = math.ceil(num_sentences / BATCH_GRAPH_SIZE)

            # Iterate batches
            for i in range(num_batches):
                start_idx = i * BATCH_GRAPH_SIZE
                end_idx = (i + 1) * BATCH_GRAPH_SIZE

                # Split sentences to batches
                _sentences: List[Sentence] = sentences[start_idx:end_idx]

                # Process batch pagerank
                _sentences, _scores = _batch_pagerank(
                    sentences=_sentences, tolerance=self.tolerance
                )

                # Append to batch results
                batched_sentences.append(_sentences)
                batched_scores.append(_scores)

            # Flatten batch results
            sentences = [item for sublist in batched_sentences for item in sublist]
            scores = [item for sublist in batched_scores for item in sublist]

        # Convert scores to rank
        ranks = sort_values(scores)

        # Insert rank of each sentence
        for sentence, rank in zip(sentences, ranks):
            sentence.update({"rank": rank})

        if sort:
            sentences = sorted(sentences, key=lambda x: x["score"], reverse=True)
        sentences = sentences[:num_sentences]

        return sentences
