import sys
import typing
import bpy.types

GenericType = typing.TypeVar("GenericType")


def action_sanitize(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None):
    ''' Make action suitable for use as a Pose Library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def apply_pose(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               pose_index: typing.Optional[typing.Any] = -1):
    ''' Apply specified Pose Library pose to the rig

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param pose_index: Pose, Index of the pose to apply (-2 for no change to pose, -1 for poselib active pose)
    :type pose_index: typing.Optional[typing.Any]
    '''

    pass


def browse_interactive(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       pose_index: typing.Optional[typing.Any] = -1):
    ''' Interactively browse poses in 3D-View

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param pose_index: Pose, Index of the pose to apply (-2 for no change to pose, -1 for poselib active pose)
    :type pose_index: typing.Optional[typing.Any]
    '''

    pass


def new(override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None):
    ''' Add New Pose Library to active Object

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def pose_add(override_context: typing.Union[typing.
                                            Dict, 'bpy.types.Context'] = None,
             execution_context: typing.Union[str, int] = None,
             undo: typing.Optional[bool] = None,
             *,
             frame: typing.Optional[typing.Any] = 1,
             name: typing.Union[str, typing.Any] = "Pose"):
    ''' Add the current Pose to the active Pose Library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param frame: Frame, Frame to store pose on
    :type frame: typing.Optional[typing.Any]
    :param name: Pose Name, Name of newly added Pose
    :type name: typing.Union[str, typing.Any]
    '''

    pass


def pose_move(override_context: typing.Union[typing.
                                             Dict, 'bpy.types.Context'] = None,
              execution_context: typing.Union[str, int] = None,
              undo: typing.Optional[bool] = None,
              *,
              pose: typing.Union[str, int, typing.Any] = '',
              direction: typing.Optional[typing.Any] = 'UP'):
    ''' Move the pose up or down in the active Pose Library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param pose: Pose, The pose to move
    :type pose: typing.Union[str, int, typing.Any]
    :param direction: Direction, Direction to move the chosen pose towards
    :type direction: typing.Optional[typing.Any]
    '''

    pass


def pose_remove(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None,
                *,
                pose: typing.Union[str, int, typing.Any] = ''):
    ''' Remove nth pose from the active Pose Library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param pose: Pose, The pose to remove
    :type pose: typing.Union[str, int, typing.Any]
    '''

    pass


def pose_rename(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None,
                *,
                name: typing.Union[str, typing.Any] = "RenamedPose",
                pose: typing.Union[str, int, typing.Any] = ''):
    ''' Rename specified pose from the active Pose Library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: New Pose Name, New name for pose
    :type name: typing.Union[str, typing.Any]
    :param pose: Pose, The pose to rename
    :type pose: typing.Union[str, int, typing.Any]
    '''

    pass


def unlink(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None):
    ''' Remove Pose Library from active Object

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass
