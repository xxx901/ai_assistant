"""
统一的日志配置。

设计原则：
1. 项目内部状态（路由决策、工具执行结果、性能耗时、异常兜底）一律用logging记录，
   不再用print——print只保留给"真正要跟用户交互的终端界面文字"用
   （比如聊天回复本身、"是否执行？(y/n)"这类确认提示），日志和用户界面是两件不同的事，
   混在一起会导致：想关掉调试信息的时候，把用户看的东西也一起关掉了。
2. 同时输出到控制台和文件——控制台方便交互时随手看，文件用于事后排查
   （不依赖"出问题的时候正好开着终端盯着看"），也是"可观测性"的基本要求。
3. 用标准的日志级别划分：
   DEBUG   - 内部细节，比如模型的原始输出、完整的执行结果字典，默认不显示，排查问题时才打开
   INFO    - 正常运行中的关键节点，比如路由决策结果、性能耗时、工具执行完成
   WARNING - 出现了异常但有兜底、不影响继续运行，比如模型输出解析失败、已经退回安全默认值
   ERROR   - 需要关注的真实错误
"""

import logging
import os
from config import LOG_LEVEL, LOG_FILE_PATH


def setup_logging():
    """整个程序启动时调用一次即可，后续各模块用 logging.getLogger(__name__) 取用。"""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # 避免重复调用setup_logging()时，同一个handler被加好几遍、导致日志重复打印
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
