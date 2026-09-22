"""
Ollama客户端封装。

设计原则：
1. 这一层只负责"跟Ollama说话"，不做任何业务判断（比如要不要重试、要不要降级）。
2. 出错时统一抛出 OllamaClientError，把requests库的底层异常屏蔽掉，
   这样上层（router、agent_loop等）不需要知道"到底是连接失败还是超时"，
   只需要 try/except OllamaClientError 就够了，具体怎么兜底由上层自己决定。
3. stream 由调用方显式传入，不做"猜测调用者是谁"这种隐式行为。
"""

import json
import requests


def is_ollama_running(base_url: str, timeout: int = 2) -> bool:
    """启动时快速探测Ollama服务是否可用，探活请求用短超时，不用等太久。"""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


class OllamaClientError(Exception):
    """统一的Ollama调用异常，屏蔽底层HTTP/网络细节。"""
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 30, num_gpu: int = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_gpu = num_gpu  # None=用Ollama默认策略；0=强制纯CPU；正整数=指定放几层到GPU

    def chat(self, messages: list, stream: bool = False, system: str = None, format_json: bool = False):
        """
        调用本地模型进行对话。

        参数：
            messages: [{"role": "user"/"assistant", "content": "..."}, ...]
                      多轮对话历史，最后一条通常是当前用户输入
            stream:   False -> 返回完整字符串（适合意图路由这种要一次性拿结果的场景）
                      True  -> 返回一个生成器，逐块yield文字（适合聊天界面，边生成边显示）
            system:   可选的系统提示词，会自动插到messages最前面
            format_json: True -> 走Ollama原生的JSON模式(强制模型输出JSON)，
                         适合结构化提炼/分类这类要稳定解析的场景。返回仍是字符串，
                         由上层自己 json.loads，失败照常兜底。

        异常：
            OllamaClientError: Ollama服务未启动 / 超时 / 返回错误状态码时抛出
        """
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}] + payload_messages

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": stream,
        }
        if format_json:
            payload["format"] = "json"
        if self.num_gpu is not None:
            payload["options"] = {"num_gpu": self.num_gpu}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout, stream=stream)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise OllamaClientError(
                "无法连接到Ollama服务，请确认已执行 `ollama serve` 且服务正在运行"
            )
        except requests.exceptions.Timeout:
            raise OllamaClientError(f"请求超时（超过 {self.timeout} 秒）")
        except requests.exceptions.HTTPError as e:
            raise OllamaClientError(f"Ollama返回错误状态: {e}")

        if stream:
            return self._stream_generator(resp)
        else:
            data = resp.json()
            return data["message"]["content"]

    def _stream_generator(self, resp):
        """把Ollama的逐行JSON流，转换成逐块的文字生成器。

        注意：读取过程本身也可能中途失败（服务器卡死、连接被重置等），
        所以这里也要用try/except包起来，统一抛出OllamaClientError，
        不能只在chat()方法里包最初那次requests.post。
        """
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
        except requests.exceptions.RequestException as e:
            raise OllamaClientError(f"流式读取中断（服务可能卡死或连接断开）: {e}")