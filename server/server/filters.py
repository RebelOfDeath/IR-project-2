from logging import Filter, LogRecord
from typing import Optional

from server.context import correlation_id


class CorrelationIdFilter(Filter):

    def __init__(self, name: str = '', default_value: Optional[str] = None):
        super().__init__(name=name)
        self.default_value = default_value

    def filter(self, record: LogRecord) -> bool:
        cid = correlation_id.get(self.default_value)
        record.correlation_id = cid
        return True
