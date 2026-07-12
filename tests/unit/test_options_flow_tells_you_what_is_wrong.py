"""A helpful error message, computed and then thrown away.

`_validate_and_convert_dhw_config` raises `vol.Invalid` with a message that says exactly what the
user got wrong:

    DHW target temperature must be between 45.0-60.0°C

`async_step_init` calls it without catching anything. An exception escaping a config-flow step is
not shown to the user - Home Assistant catches it, logs a traceback, and renders the generic
**"Unknown error occurred"**. So the sentence above is written, and never read. The user is told
that something failed, not what, and their input is gone.

Home Assistant's own pattern is to collect the problem into an `errors` dict and re-show the form
with the message attached to the field. That is what the config flow already does elsewhere in this
integration; the options flow does not.

Nothing here reaches the heat pump. It reaches the person trying to configure one.
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
