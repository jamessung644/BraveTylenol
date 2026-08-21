from lunit_hackathon.deadline import RequestDeadline


def test_stage_timeout_preserves_reserve():
    now = iter([10.0, 30.0])
    deadline = RequestDeadline.start(total_seconds=100.0, clock=lambda: next(now))

    assert deadline.stage_timeout(75.0, reserve_seconds=25.0) == 55.0


def test_stage_timeout_never_becomes_negative():
    now = iter([10.0, 100.0])
    deadline = RequestDeadline.start(total_seconds=50.0, clock=lambda: next(now))

    assert deadline.stage_timeout(75.0, reserve_seconds=25.0) == 0.0


def test_can_spend_checks_the_remaining_budget_boundary():
    now = iter([10.0, 10.0, 10.1])
    deadline = RequestDeadline.start(total_seconds=50.0, clock=lambda: next(now))

    assert deadline.can_spend(50.0)
    assert not deadline.can_spend(50.0)
