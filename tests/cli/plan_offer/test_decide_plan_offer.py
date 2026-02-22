from __future__ import annotations

import pytest

from tests.cli.plan_offer.adapters.fake_whoami_gateway import FakeWhoAmIGateway
from vibe.cli.plan_offer.decide_plan_offer import (
    PlanOfferAction,
    PlanType,
    decide_plan_offer,
)
from vibe.cli.plan_offer.ports.whoami_gateway import WhoAmIResponse


@pytest.mark.asyncio
async def test_always_returns_none() -> None:
    gateway = FakeWhoAmIGateway(
        WhoAmIResponse(
            is_pro_plan=False,
            advertise_pro_plan=False,
            prompt_switching_to_pro_plan=False,
        )
    )
    action, plan_type = await decide_plan_offer("api-key", gateway)

    assert action is PlanOfferAction.NONE
    assert plan_type is PlanType.UNKNOWN


@pytest.mark.asyncio
async def test_always_returns_none_without_api_key() -> None:
    gateway = FakeWhoAmIGateway(
        WhoAmIResponse(
            is_pro_plan=True, advertise_pro_plan=True, prompt_switching_to_pro_plan=True
        )
    )
    action, plan_type = await decide_plan_offer("", gateway)

    assert action is PlanOfferAction.NONE
    assert plan_type is PlanType.UNKNOWN
