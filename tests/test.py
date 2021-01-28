# -*- coding: utf-8 -*-

import unittest

from typing import List, Tuple

from textrankr import TextRank

from .tokenizers import OktTokenizer


class TestTextRank(unittest.TestCase):
    def setUp(self) -> None:
        self.text: str = '트위터, "정보당국에 데이터 분석자료 팔지 않겠다". 트위터가 수많은 트윗을 분석해 정보를 판매하는 서비스를 미국 정보당국에는 제공하지 않기로 했다. 월스트리트저널은 미국 정보당국 관계자 등을 인용해 데이터마이너(Dataminer)가 정보당국에 대한 서비스는 중단하기로 했다고 9일(현지시간) 보도했다. 트위터가 5% 지분을 가진 데이터마이너는 소셜미디어상 자료를 분석해 고객이 의사결정을 하도록 정보를 제공하는 기업이다. 트위터에 올라오는 트윗에 실시간으로 접근해 분석한 자료를 고객에게 팔 수 있는 독점권을 갖고 있다. 정보당국은 이 회사로부터 구매한 자료로 테러나 정치적 불안정 등과 관련된 정보를 획득했다. 이 회사가 정보당국에 서비스를 판매하지 않기로 한 것은 트위터의 결정인 것으로 알려졌다. 데이터마이너 경영진은 최근 “트위터가 정보당국에 서비스하는 것을 원치 않는다”고 밝혔다고 이 신문은 전했다. 트위터도 성명을 내고 “정보당국 감시용으로 데이터를 팔지 않는 것은 트위터의 오래된 정책”이라며 “트위터 자료는 대체로 공개적이고 미국 정부도 다른 사용자처럼 공개된 어카운트를 살펴볼 수 있다”고 해명했다. 그러나 이는 이 회사가 2년 동안 정보당국에 서비스를 제공해 온 데 대해서는 타당한 설명이 되지 않는다. 트위터의 이번 결정은 미국의 정보기술(IT)기업과 정보당국 간 갈등의 연장 선상에서 이뤄진 것으로 여겨지고 있다. IT기업은 이용자 프라이버시에 무게 중심을 두는 데 비해 정보당국은 공공안전을 우선시해 차이가 있었다. 특히 애플은 캘리포니아 주 샌버너디노 총격범의 아이폰에 저장된 정보를 보겠다며 데이터 잠금장치 해제를 요구하는 미 연방수사국(FBI)과 소송까지 진행했다. 정보당국 고위 관계자도 “트위터가 정보당국과 너무 가까워 보이는 것을 우려하는 것 같다”고 말했다. 데이터마이너는 금융기관이나, 언론사 등 정보당국을 제외한 고객에 대한 서비스는 계속할 계획이다. .'
        self.tokenizer: OktTokenizer = OktTokenizer()
        self.textrank: TextRank = TextRank(self.tokenizer)

    def test_ranked(self) -> None:
        summaries: List[str] = self.textrank.summarize(self.text, 3, verbose=False)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(summaries[0], '트위터, "정보당국에 데이터 분석자료 팔지 않겠다"')

    def test_verbose(self) -> None:
        summaries: str = self.textrank.summarize(self.text, 1, verbose=True)
        self.assertEqual(summaries, '트위터, "정보당국에 데이터 분석자료 팔지 않겠다"')

    def test_rank(self) -> None:
        summaries: List[Tuple[str, float]] = self.textrank.rank(
            self.text, 3, verbose=False, sort=True
        )
        self.assertEqual(len(summaries), 3)
        self.assertEqual(summaries[0]["sentence"], '트위터, "정보당국에 데이터 분석자료 팔지 않겠다"')
        self.assertEqual(summaries[0]["rank"], 0)


if __name__ == "__main__":
    unittest.main()
