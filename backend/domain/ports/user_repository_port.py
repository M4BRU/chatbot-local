"""Port UserRepository — pour usage futur (swap SQLAlchemy → autre adapter)."""

import uuid
from abc import ABC, abstractmethod

from backend.db.models import UserTable


class UserRepositoryPort(ABC):
    @abstractmethod
    async def get_by_id(self, user_id: uuid.UUID) -> UserTable | None:
        raise NotImplementedError
