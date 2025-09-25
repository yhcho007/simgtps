# backend/app/core/tools.py
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def run(self, **kwargs) -> Any:
        raise NotImplementedError(f"Tool '{self.name}' must implement run method.")

class InternalAccountAPI(Tool):
    def __init__(self):
        super().__init__("InternalAccountAPI", "은행 내부 시스템에서 고객 계좌 정보를 조회합니다 (예: 휴면 계좌 잔고, 특정 기간 거래 내역).")

    async def run(self, customer_id: str, query: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"InternalAccountAPI 호출: 고객ID={customer_id}, 쿼리='{query}'")
        # --- 실제 내부망 API 연동 로직 구현 (가상 응답) ---
        if "휴면 계좌 잔고" in query:
            return {"status": "success", "result": f"고객 {customer_id}님의 휴면 계좌에 12,345원 남아있습니다."}
        elif "거래 내역" in query:
            target_person = kwargs.get("target_person", "특정인")
            period = kwargs.get("period", "3년")
            return {"status": "success", "result": f"고객 {customer_id}님의 {period} 이내 {target_person}과의 거래 내역을 조회했습니다. 상세 내역은 보안상 영업점에서 확인해주세요."}
        else:
            return {"status": "failure", "message": "지원하지 않는 계좌 조회 쿼리입니다."}

class FinancialProductRecommendationAPI(Tool):
    def __init__(self):
        super().__init__("FinancialProductRecommendationAPI", "고객의 정보와 요청을 기반으로 맞춤형 금융 상품 (예: 대출, 카드)을 추천합니다.")

    async def run(self, customer_id: str, criteria: str, product_type: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"FinancialProductRecommendationAPI 호출: 고객ID={customer_id}, 기준='{criteria}', 상품유형='{product_type}'")
        # --- 실제 금융 상품 추천 시스템 연동 로직 구현 (가상 응답) ---
        if product_type == "대출":
            return {"status": "success", "result": f"고객님께는 '행복드림 주택담보대출' (연 최저 3.5%) 또는 '사이다 신용대출' (연 최저 4.0%)을 추천합니다."}
        elif product_type == "카드":
            return {"status": "success", "result": f"고객님 소비패턴에 맞춰 '포인트팡팡 카드' 또는 '온라인쇼핑 할인 카드'를 추천합니다."}
        else:
            return {"status": "failure", "message": "지원하지 않는 상품 유형입니다."}

# --- 다른 Agent 연동 예시 ---
# 다른 AI Agent가 특정 작업을 처리하고 결과를 반환한다고 가정
class ExternalCreditScoreAgent(Tool):
    def __init__(self):
        super().__init__("ExternalCreditScoreAgent", "외부 신용 평가 Agent를 호출하여 고객의 실시간 신용 상태를 조회합니다.")

    async def run(self, customer_id: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"ExternalCreditScoreAgent 호출: 고객ID={customer_id}")
        # --- 외부 Agent 연동 로직 구현 (가상 응답) ---
        return {"status": "success", "result": f"고객 {customer_id}님의 현재 신용점수는 850점 (우수) 입니다."}

# --- 멀티모달 응답용 리소스 검색 툴 ---
class MultimodalResourceLookup(Tool):
    def __init__(self):
        super().__init__("MultimodalResourceLookup", "텍스트, PDF 문서, 이미지와 같은 멀티모달 리소스를 검색합니다.")

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"MultimodalResourceLookup 호출: 쿼리='{query}'")
        if "주택담보대출" in query and "가이드" in query:
            return {"status": "success", "resource_type": "pdf", "path": "bank_loan_guide.pdf", "text_summary": "주택담보대출 가이드 PDF를 찾았습니다. 대출 절차, 필요 서류 등이 상세히 설명되어 있습니다."}
        elif "은행 카드" in query and "혜택" in query:
            return {"status": "success", "resource_type": "image", "path": "bank_card_benefits.png", "text_summary": "은행 카드 혜택 요약 이미지를 찾았습니다. 주요 카드별 혜택이 보기 쉽게 정리되어 있습니다."}
        else:
            return {"status": "failure", "message": "요청하신 멀티모달 리소스를 찾을 수 없습니다."}

# 사용 가능한 모든 툴들을 리스트로 묶어 에이전트가 선택할 수 있도록 합니다.
def get_all_tools() -> List[Tool]:
    return [
        InternalAccountAPI(),
        FinancialProductRecommendationAPI(),
        ExternalCreditScoreAgent(),
        MultimodalResourceLookup()
    ]