from __future__ import annotations

from enum import StrEnum
import os

from vibe.cli.plan_offer.ports.whoami_gateway import WhoAmIGateway
from vibe.core.config import DEFAULT_MISTRAL_API_ENV_KEY, Backend, ProviderConfig


class PlanOfferAction(StrEnum):
    NONE = "none"


class PlanType(StrEnum):
    UNKNOWN = "unknown"
    FREE = "free"


async def decide_plan_offer(
    api_key: str | None, gateway: WhoAmIGateway
) -> tuple[PlanOfferAction, PlanType]:
    return PlanOfferAction.NONE, PlanType.UNKNOWN


def plan_offer_cta(action: PlanOfferAction) -> str | None:
    return None


def resolve_api_key_for_plan(provider: ProviderConfig) -> str | None:
    return None
