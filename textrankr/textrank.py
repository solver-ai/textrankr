from typing import List
from typing import Dict
from typing import Callable
from typing import Union

from networkx import Graph
from networkx import pagerank_scipy as pagerank

from .sentence import Sentence

from .utils import parse_text_into_sentences, parse_candidates_to_sentences
from .utils import build_sentence_graph
from .utils import sort_values


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

        # build graph
        graph: Graph = build_sentence_graph(sentences, tolerance=self.tolerance)

        # run pagerank
        pageranks: Dict[Sentence, float] = pagerank(graph, weight="weight")

        # get top-k sentences
        sentences = [
            {"sentence": k.text, "index": k.index, "score": v}
            for i, (k, v) in enumerate(pageranks.items())
        ]
        scores = list(pageranks.values())
        ranks = sort_values(scores)

        # Insert rank of each sentence
        for sentence, rank in zip(sentences, ranks):
            sentence.update({"rank": rank})

        if sort:
            sentences = sorted(sentences, key=lambda x: x["score"], reverse=True)
        sentences = sentences[:num_sentences]

        return sentences
