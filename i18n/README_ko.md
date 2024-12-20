# 🧩 BasketPuzzle Project

## 🌎 Translation Docs
- [한국어](https://github.com/BasketPuzzle/BasketPuzzle-FE/blob/main/i18n/README_ko.md)
- [English](https://github.com/BasketPuzzle/BasketPuzzle-FE/blob/main/i18n/README_en.md)

## 문제 정의 및 분석
현대 마케팅에서 소비자가 장바구니에 무엇을 담는지 제품 패턴을 분석하는 것은 매우 중요한 과제이다.
이를 위해 소비자의 소비 패턴을 이해하고, 함께 구매되는 제품을 식별하는 것은 필수적이다.
따라서 본 프로젝트의 목표는, 소비자의 장바구니 패턴을 분석하여 어떠한 제품 조합이 성공적인 거래로 이어지는지 확인하고, 이를 통해 효과적인 마케팅 전략을 수립하는 것이다.
그 뿐만 아니라 이미 이와 비슷한 기존의 연구와 차별화하를 위해, 우리는 구매 이력을 바탕으로 소매 및 도매 고객으로 소비자를 분류하고,
각 그룹에 맞춤형 패키지 제품을 추천하는 알고리즘을 구현하고자 한다.
이러한 접근 방식은 실제 비즈니스 전략에 더욱 더 가치 있는 통찰력을 제공할 것이다.


## 프로젝트 목표
BasketPuzzle의 목표는 소비자의 구매 이력을 분석하여 제품 간의 패턴을 발견하고, 이를 정량화된 데이터로 표현하는 것이다.
우리는 API 파이프라인을 구축하고, 분석 결과를 웹으로 시각화하여 사용자들이 쉽게 접근할 수 있도록 할 것이다.
이를 통해 판매자는 보다 효과적인 마케팅 전략을 개발하고 고객 만족도를 높일 수 있으며,
소비자는 맞춤형 추천을 통해 개선된 쇼핑 경험을 즐길 수 있을 것이다.

## 주요 기능
1. 장바구니 데이터 차트 시각화
2. 쇼핑 트렌드 목록 차트 제공
3. 특정 상품명 검색에 따른 정보 출력. (판매량, 관련 상품 TOP3)
4. 사용자 입력을 통한 구매 관련 제품 분석 및 추천

## 개발 도구 및 언어
쇼핑 데이터 분석은 Python 기반의 Jupyter Notebook 환경에서 진행된다.<br>
데이터 분석, 전처리, 계산을 위해 Pandas, NumPy, Matplotlib 등의 Python 라이브러리들을 활용할 것이다.<br>
또한 연관 규칙 학습을 위해 Apriori알고리즘을 사용하였다.<br>
마지막으로, 백엔드 API는 Flask를 사용하여 구축하고,<br>
웹에서 분석 결과를 시각화하기 위해 HTML, CSS, JavaScript를 활용할 계획이다. 

## ReadTheDocs
[https://basketpuzzle-fe-docs.readthedocs.io/en/latest/](https://basketpuzzle-fe-docs.readthedocs.io/en/latest/)

## Jekyll Page
[https://basketpuzzle.github.io/BasketPuzzle-FE/](https://basketpuzzle.github.io/BasketPuzzle-FE/)

## Service Page
https://basketpuzzle.github.io/basketpuzzlehosting/
