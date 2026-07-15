"""The options flow must surface its own validation message, not throw it away.

`_validate_and_convert_dhw_config` raises `vol.Invalid` with a message naming the field and its
permitted range. `async_step_init` must catch it and re-show the form with the message in an
`errors` dict - an exception left to escape a config-flow step renders as HA's generic "Unknown
error occurred", so the user is told that something failed but not what, and their input is gone.
"""

from __future__ import annotations

import inspect

import pytest
import voluptuous as vol

from custom_components.effektguard.options import EffektGuardOptionsFlow


def test_the_validator_still_rejects_an_out_of_range_target():
    """The precondition. If this stops raising, the rest of the file is about nothing."""
    flow = EffektGuardOptionsFlow()

    with pytest.raises(vol.Invalid):
        flow._validate_and_convert_dhw_config({"dhw_target_temp": 95.0})


def test_the_step_does_not_let_the_error_escape_as_unknown_error():
    """An unhandled exception in a flow step renders as "Unknown error occurred"."""
    source = inspect.getsource(EffektGuardOptionsFlow.async_step_init)

    assert "vol.Invalid" in source, (
        "async_step_init calls _validate_and_convert_dhw_config, which raises vol.Invalid with a "
        "message naming the field and the permitted range - and does not catch it. Home Assistant "
        "turns an escaped exception into 'Unknown error occurred', so the message is never seen "
        "and the user's input is discarded."
    )


def test_the_step_re_shows_the_form_with_the_message_on_it():
    """Catching it is only half the job: the user has to be told, on the field."""
    source = inspect.getsource(EffektGuardOptionsFlow.async_step_init)

    assert "errors" in source, (
        "async_step_init must collect the validation failure into an `errors` dict and pass it to "
        "async_show_form, so the message lands on the form the user is looking at."
    )
