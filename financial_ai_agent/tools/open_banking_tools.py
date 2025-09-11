from langchain.tools import tool
from api_clients.open_banking import get_accounts as get_accounts_client
from api_clients.open_banking import get_loans as get_loans_client

@tool
def get_accounts():
    """사용자의 모든 은행 계좌 정보를 조회합니다. 잔액, 은행 이름 등을 확인할 수 있습니다."""
    return get_accounts_client()

@tool
def get_loans():
    """사용자의 모든 대출 정보를 조회합니다. 대출 유형, 잔액, 월 상환액 등을 확인할 수 있습니다."""
    return get_loans_client()