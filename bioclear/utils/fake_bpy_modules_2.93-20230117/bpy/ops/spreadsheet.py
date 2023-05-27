import sys
import typing
import bpy.types

GenericType = typing.TypeVar("GenericType")


def toggle_pin(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None):
    ''' Turn on or off pinning :file: `startup/bl_operators/spreadsheet.py\:34 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/spreadsheet.py$34>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass
