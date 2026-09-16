import requests

def is_ollama_running(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        # 判断状态码，返回True/False
        if resp.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

"""官方答案：
def is_ollama_running(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
        """