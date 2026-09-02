from backend.follow_object import FOLLOW_SEQUENCE, FollowObjectStateMachine


def test_follow_object_is_opt_in_and_uses_dedicated_fist():
    machine = FollowObjectStateMachine(timeout_seconds=4.0, hold_seconds=0.2)
    assert machine.observe("palm", stable=True, now=0.0).state == "inactive"
    started = machine.start(now=0.0)
    assert started.expected_gesture == "palm"
    assert FOLLOW_SEQUENCE == ("palm", "fist", "palm")

    machine.observe("palm", stable=True, now=0.0)
    first = machine.observe("palm", stable=True, now=0.21)
    assert first.state == "wait_fist"
    assert first.message == "Grab the object"

    # RPS rock remains Pause Video and must never substitute for HaGRID fist.
    rock = machine.observe("rock", stable=True, now=0.3)
    assert rock.state == "wait_fist"
    assert rock.step_index == 1

    machine.observe("fist", stable=True, now=0.4)
    second = machine.observe("fist", stable=True, now=0.61)
    assert second.state == "wait_final_palm"
    assert second.message == "Release the object"

    machine.observe("palm", stable=True, now=0.8)
    complete = machine.observe("palm", stable=True, now=1.01)
    assert complete.completed is True
    assert complete.message == "Follow the object successfully done"
    assert machine.state == "inactive"


def test_step_timeout_restarts_from_palm_without_closing_session():
    machine = FollowObjectStateMachine(timeout_seconds=2.0, hold_seconds=0.0)
    machine.start(now=0.0)
    accepted = machine.observe("palm", stable=True, now=0.0)
    assert accepted.state == "wait_fist"
    update = machine.observe("fist", stable=False, now=3.0)
    assert update.reset is True
    assert update.active is True
    assert update.state == "wait_palm"


def test_wrong_pose_waits_for_expected_pose():
    machine = FollowObjectStateMachine(timeout_seconds=4.0, hold_seconds=0.0)
    machine.start(now=0.0)
    machine.observe("palm", stable=True, now=0.0)
    update = machine.observe("peace", stable=True, now=0.4)
    assert update.reset is False
    assert update.state == "wait_fist"
    assert update.expected_gesture == "fist"
