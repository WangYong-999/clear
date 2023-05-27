import sys
import typing
import bpy.types
import bpy.ops.transform

GenericType = typing.TypeVar("GenericType")


def bake(override_context: typing.Union[typing.
                                        Dict, 'bpy.types.Context'] = None,
         execution_context: typing.Union[str, int] = None,
         undo: typing.Optional[bool] = None):
    ''' Bake selected F-Curves to a set of sampled points defining a similar curve

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def clean(override_context: typing.Union[typing.
                                         Dict, 'bpy.types.Context'] = None,
          execution_context: typing.Union[str, int] = None,
          undo: typing.Optional[bool] = None,
          *,
          threshold: typing.Optional[typing.Any] = 0.001,
          channels: typing.Union[bool, typing.Any] = False):
    ''' Simplify F-Curves by removing closely spaced keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param threshold: Threshold
    :type threshold: typing.Optional[typing.Any]
    :param channels: Channels
    :type channels: typing.Union[bool, typing.Any]
    '''

    pass


def click_insert(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 frame: typing.Optional[typing.Any] = 1.0,
                 value: typing.Optional[typing.Any] = 1.0,
                 extend: typing.Union[bool, typing.Any] = False):
    ''' Insert new keyframe at the cursor position for the active F-Curve

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param frame: Frame Number, Frame to insert keyframe on
    :type frame: typing.Optional[typing.Any]
    :param value: Value, Value for keyframe on
    :type value: typing.Optional[typing.Any]
    :param extend: Extend, Extend selection instead of deselecting everything first
    :type extend: typing.Union[bool, typing.Any]
    '''

    pass


def clickselect(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        wait_to_deselect_others: typing.Union[bool, typing.Any] = False,
        mouse_x: typing.Optional[typing.Any] = 0,
        mouse_y: typing.Optional[typing.Any] = 0,
        extend: typing.Union[bool, typing.Any] = False,
        deselect_all: typing.Union[bool, typing.Any] = False,
        column: typing.Union[bool, typing.Any] = False,
        curves: typing.Union[bool, typing.Any] = False):
    ''' Select keyframes by clicking on them

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param wait_to_deselect_others: Wait to Deselect Others
    :type wait_to_deselect_others: typing.Union[bool, typing.Any]
    :param mouse_x: Mouse X
    :type mouse_x: typing.Optional[typing.Any]
    :param mouse_y: Mouse Y
    :type mouse_y: typing.Optional[typing.Any]
    :param extend: Extend Select, Toggle keyframe selection instead of leaving newly selected keyframes only
    :type extend: typing.Union[bool, typing.Any]
    :param deselect_all: Deselect On Nothing, Deselect all when nothing under the cursor
    :type deselect_all: typing.Union[bool, typing.Any]
    :param column: Column Select, Select all keyframes that occur on the same frame as the one under the mouse
    :type column: typing.Union[bool, typing.Any]
    :param curves: Only Curves, Select all the keyframes in the curve
    :type curves: typing.Union[bool, typing.Any]
    '''

    pass


def copy(override_context: typing.Union[typing.
                                        Dict, 'bpy.types.Context'] = None,
         execution_context: typing.Union[str, int] = None,
         undo: typing.Optional[bool] = None):
    ''' Copy selected keyframes to the copy/paste buffer

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def cursor_set(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               frame: typing.Optional[typing.Any] = 0.0,
               value: typing.Optional[typing.Any] = 0.0):
    ''' Interactively set the current frame and value cursor

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param frame: Frame
    :type frame: typing.Optional[typing.Any]
    :param value: Value
    :type value: typing.Optional[typing.Any]
    '''

    pass


def decimate(override_context: typing.Union[typing.
                                            Dict, 'bpy.types.Context'] = None,
             execution_context: typing.Union[str, int] = None,
             undo: typing.Optional[bool] = None,
             *,
             mode: typing.Optional[typing.Any] = 'RATIO',
             remove_ratio: typing.Optional[typing.Any] = 0.333333,
             remove_error_margin: typing.Optional[typing.Any] = 0.0):
    ''' Decimate F-Curves by removing keyframes that influence the curve shape the least

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param mode: Mode, Which mode to use for decimation * ``RATIO`` Ratio, Use a percentage to specify how many keyframes you want to remove. * ``ERROR`` Error Margin, Use an error margin to specify how much the curve is allowed to deviate from the original path.
    :type mode: typing.Optional[typing.Any]
    :param remove_ratio: Remove, The ratio of remaining keyframes after the operation
    :type remove_ratio: typing.Optional[typing.Any]
    :param remove_error_margin: Max Error Margin, How much the new decimated curve is allowed to deviate from the original
    :type remove_error_margin: typing.Optional[typing.Any]
    '''

    pass


def delete(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None,
           *,
           confirm: typing.Union[bool, typing.Any] = True):
    ''' Remove all selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param confirm: Confirm, Prompt for confirmation
    :type confirm: typing.Union[bool, typing.Any]
    '''

    pass


def driver_delete_invalid(override_context: typing.
                          Union[typing.Dict, 'bpy.types.Context'] = None,
                          execution_context: typing.Union[str, int] = None,
                          undo: typing.Optional[bool] = None):
    ''' Delete all visible drivers considered invalid

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def driver_variables_copy(override_context: typing.
                          Union[typing.Dict, 'bpy.types.Context'] = None,
                          execution_context: typing.Union[str, int] = None,
                          undo: typing.Optional[bool] = None):
    ''' Copy the driver variables of the active driver

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def driver_variables_paste(override_context: typing.
                           Union[typing.Dict, 'bpy.types.Context'] = None,
                           execution_context: typing.Union[str, int] = None,
                           undo: typing.Optional[bool] = None,
                           *,
                           replace: typing.Union[bool, typing.Any] = False):
    ''' Add copied driver variables to the active driver

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param replace: Replace Existing, Replace existing driver variables, instead of just appending to the end of the existing list
    :type replace: typing.Union[bool, typing.Any]
    '''

    pass


def duplicate(override_context: typing.Union[typing.
                                             Dict, 'bpy.types.Context'] = None,
              execution_context: typing.Union[str, int] = None,
              undo: typing.Optional[bool] = None,
              *,
              mode: typing.Optional[typing.Any] = 'TRANSLATION'):
    ''' Make a copy of all selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param mode: Mode
    :type mode: typing.Optional[typing.Any]
    '''

    pass


def duplicate_move(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   GRAPH_OT_duplicate: typing.Optional['duplicate'] = None,
                   TRANSFORM_OT_transform: typing.
                   Optional['bpy.ops.transform.transform'] = None):
    ''' Make a copy of all selected keyframes and move them

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param GRAPH_OT_duplicate: Duplicate Keyframes, Make a copy of all selected keyframes
    :type GRAPH_OT_duplicate: typing.Optional['duplicate']
    :param TRANSFORM_OT_transform: Transform, Transform selected items by mode type
    :type TRANSFORM_OT_transform: typing.Optional['bpy.ops.transform.transform']
    '''

    pass


def easing_type(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None,
                *,
                type: typing.Optional[typing.Any] = 'AUTO'):
    ''' Set easing type for the F-Curve segments starting from the selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``AUTO`` Automatic Easing, Easing type is chosen automatically based on what the type of interpolation used (e.g. Ease In for transitional types, and Ease Out for dynamic effects). * ``EASE_IN`` Ease In, Only on the end closest to the next keyframe. * ``EASE_OUT`` Ease Out, Only on the end closest to the first keyframe. * ``EASE_IN_OUT`` Ease In and Out, Segment between both keyframes.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def euler_filter(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None):
    ''' Fix large jumps and flips in the selected Euler Rotation F-Curves arising from rotation values being clipped when baking physics

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def extrapolation_type(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       type: typing.Optional[typing.Any] = 'CONSTANT'):
    ''' Set extrapolation mode for selected F-Curves

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``CONSTANT`` Constant Extrapolation, Values on endpoint keyframes are held. * ``LINEAR`` Linear Extrapolation, Straight-line slope of end segments are extended past the endpoint keyframes. * ``MAKE_CYCLIC`` Make Cyclic (F-Modifier), Add Cycles F-Modifier if one doesn't exist already. * ``CLEAR_CYCLIC`` Clear Cyclic (F-Modifier), Remove Cycles F-Modifier if not needed anymore.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def fmodifier_add(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  type: typing.Optional[typing.Any] = 'NULL',
                  only_active: typing.Union[bool, typing.Any] = True):
    ''' Add F-Modifier to the active/selected F-Curves

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``NULL`` Invalid. * ``GENERATOR`` Generator, Generate a curve using a factorized or expanded polynomial. * ``FNGENERATOR`` Built-In Function, Generate a curve using standard math functions such as sin and cos. * ``ENVELOPE`` Envelope, Reshape F-Curve values, e.g. change amplitude of movements. * ``CYCLES`` Cycles, Cyclic extend/repeat keyframe sequence. * ``NOISE`` Noise, Add pseudo-random noise on top of F-Curves. * ``LIMITS`` Limits, Restrict maximum and minimum values of F-Curve. * ``STEPPED`` Stepped Interpolation, Snap values to nearest grid step, e.g. for a stop-motion look.
    :type type: typing.Optional[typing.Any]
    :param only_active: Only Active, Only add F-Modifier to active F-Curve
    :type only_active: typing.Union[bool, typing.Any]
    '''

    pass


def fmodifier_copy(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None):
    ''' Copy the F-Modifier(s) of the active F-Curve

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def fmodifier_paste(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    only_active: typing.Union[bool, typing.Any] = False,
                    replace: typing.Union[bool, typing.Any] = False):
    ''' Add copied F-Modifiers to the selected F-Curves

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param only_active: Only Active, Only paste F-Modifiers on active F-Curve
    :type only_active: typing.Union[bool, typing.Any]
    :param replace: Replace Existing, Replace existing F-Modifiers, instead of just appending to the end of the existing list
    :type replace: typing.Union[bool, typing.Any]
    '''

    pass


def frame_jump(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None):
    ''' Place the cursor on the midpoint of selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def ghost_curves_clear(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None):
    ''' Clear F-Curve snapshots (Ghosts) for active Graph Editor

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def ghost_curves_create(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None):
    ''' Create snapshot (Ghosts) of selected F-Curves as background aid for active Graph Editor

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def handle_type(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None,
                *,
                type: typing.Optional[typing.Any] = 'FREE'):
    ''' Set type of handle for selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``FREE`` Free, Completely independent manually set handle. * ``ALIGNED`` Aligned, Manually set handle with rotation locked together with its pair. * ``VECTOR`` Vector, Automatic handles that create straight lines. * ``AUTO`` Automatic, Automatic handles that create smooth curves. * ``AUTO_CLAMPED`` Auto Clamped, Automatic handles that create smooth curves which only change direction at keyframes.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def hide(override_context: typing.Union[typing.
                                        Dict, 'bpy.types.Context'] = None,
         execution_context: typing.Union[str, int] = None,
         undo: typing.Optional[bool] = None,
         *,
         unselected: typing.Union[bool, typing.Any] = False):
    ''' Hide selected curves from Graph Editor view

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param unselected: Unselected, Hide unselected rather than selected curves
    :type unselected: typing.Union[bool, typing.Any]
    '''

    pass


def interpolation_type(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       type: typing.Optional[typing.Any] = 'CONSTANT'):
    ''' Set interpolation mode for the F-Curve segments starting from the selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``CONSTANT`` Constant, No interpolation, value of A gets held until B is encountered. * ``LINEAR`` Linear, Straight-line interpolation between A and B (i.e. no ease in/out). * ``BEZIER`` Bezier, Smooth interpolation between A and B, with some control over curve shape. * ``SINE`` Sinusoidal, Sinusoidal easing (weakest, almost linear but with a slight curvature). * ``QUAD`` Quadratic, Quadratic easing. * ``CUBIC`` Cubic, Cubic easing. * ``QUART`` Quartic, Quartic easing. * ``QUINT`` Quintic, Quintic easing. * ``EXPO`` Exponential, Exponential easing (dramatic). * ``CIRC`` Circular, Circular easing (strongest and most dynamic). * ``BACK`` Back, Cubic easing with overshoot and settle. * ``BOUNCE`` Bounce, Exponentially decaying parabolic bounce, like when objects collide. * ``ELASTIC`` Elastic, Exponentially decaying sine wave, like an elastic band.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def keyframe_insert(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    type: typing.Optional[typing.Any] = 'ALL'):
    ''' Insert keyframes for the specified channels

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``ALL`` All Channels, Insert a keyframe on all visible and editable F-Curves using each curve's current value. * ``SEL`` Only Selected Channels, Insert a keyframe on selected F-Curves using each curve's current value. * ``CURSOR_ACTIVE`` Active Channels at Cursor, Insert a keyframe for the active F-Curve at the cursor point. * ``CURSOR_SEL`` Selected Channels at Cursor, Insert a keyframe for selected F-Curves at the cursor point.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def mirror(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None,
           *,
           type: typing.Optional[typing.Any] = 'CFRA'):
    ''' Flip selected keyframes over the selected mirror line

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``CFRA`` By Times Over Current Frame, Flip times of selected keyframes using the current frame as the mirror line. * ``VALUE`` By Values Over Cursor Value, Flip values of selected keyframes using the cursor value (Y/Horizontal component) as the mirror line. * ``YAXIS`` By Times Over Zero Time, Flip times of selected keyframes, effectively reversing the order they appear in. * ``XAXIS`` By Values Over Zero Value, Flip values of selected keyframes (i.e. negative values become positive, and vice versa). * ``MARKER`` By Times Over First Selected Marker, Flip times of selected keyframes using the first selected marker as the reference point.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def paste(override_context: typing.Union[typing.
                                         Dict, 'bpy.types.Context'] = None,
          execution_context: typing.Union[str, int] = None,
          undo: typing.Optional[bool] = None,
          *,
          offset: typing.Optional[typing.Any] = 'START',
          merge: typing.Optional[typing.Any] = 'MIX',
          flipped: typing.Union[bool, typing.Any] = False):
    ''' Paste keyframes from copy/paste buffer for the selected channels, starting on the current frame

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param offset: Offset, Paste time offset of keys * ``START`` Frame Start, Paste keys starting at current frame. * ``END`` Frame End, Paste keys ending at current frame. * ``RELATIVE`` Frame Relative, Paste keys relative to the current frame when copying. * ``NONE`` No Offset, Paste keys from original time.
    :type offset: typing.Optional[typing.Any]
    :param merge: Type, Method of merging pasted keys and existing * ``MIX`` Mix, Overlay existing with new keys. * ``OVER_ALL`` Overwrite All, Replace all keys. * ``OVER_RANGE`` Overwrite Range, Overwrite keys in pasted range. * ``OVER_RANGE_ALL`` Overwrite Entire Range, Overwrite keys in pasted range, using the range of all copied keys.
    :type merge: typing.Optional[typing.Any]
    :param flipped: Flipped, Paste keyframes from mirrored bones if they exist
    :type flipped: typing.Union[bool, typing.Any]
    '''

    pass


def previewrange_set(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None):
    ''' Automatically set Preview Range based on range of keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def reveal(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None,
           *,
           select: typing.Union[bool, typing.Any] = True):
    ''' Make previously hidden curves visible again in Graph Editor view

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param select: Select
    :type select: typing.Union[bool, typing.Any]
    '''

    pass


def sample(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None):
    ''' Add keyframes on every frame between the selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def select_all(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               action: typing.Optional[typing.Any] = 'TOGGLE'):
    ''' Toggle selection of all keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param action: Action, Selection action to execute * ``TOGGLE`` Toggle, Toggle selection for all elements. * ``SELECT`` Select, Select all elements. * ``DESELECT`` Deselect, Deselect all elements. * ``INVERT`` Invert, Invert selection of all elements.
    :type action: typing.Optional[typing.Any]
    '''

    pass


def select_box(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               axis_range: typing.Union[bool, typing.Any] = False,
               include_handles: typing.Union[bool, typing.Any] = True,
               tweak: typing.Union[bool, typing.Any] = False,
               xmin: typing.Optional[typing.Any] = 0,
               xmax: typing.Optional[typing.Any] = 0,
               ymin: typing.Optional[typing.Any] = 0,
               ymax: typing.Optional[typing.Any] = 0,
               wait_for_input: typing.Union[bool, typing.Any] = True,
               mode: typing.Optional[typing.Any] = 'SET'):
    ''' Select all keyframes within the specified region

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param axis_range: Axis Range
    :type axis_range: typing.Union[bool, typing.Any]
    :param include_handles: Include Handles, Are handles tested individually against the selection criteria
    :type include_handles: typing.Union[bool, typing.Any]
    :param tweak: Tweak, Operator has been activated using a tweak event
    :type tweak: typing.Union[bool, typing.Any]
    :param xmin: X Min
    :type xmin: typing.Optional[typing.Any]
    :param xmax: X Max
    :type xmax: typing.Optional[typing.Any]
    :param ymin: Y Min
    :type ymin: typing.Optional[typing.Any]
    :param ymax: Y Max
    :type ymax: typing.Optional[typing.Any]
    :param wait_for_input: Wait for Input
    :type wait_for_input: typing.Union[bool, typing.Any]
    :param mode: Mode * ``SET`` Set, Set a new selection. * ``ADD`` Extend, Extend existing selection. * ``SUB`` Subtract, Subtract existing selection.
    :type mode: typing.Optional[typing.Any]
    '''

    pass


def select_circle(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  x: typing.Optional[typing.Any] = 0,
                  y: typing.Optional[typing.Any] = 0,
                  radius: typing.Optional[typing.Any] = 25,
                  wait_for_input: typing.Union[bool, typing.Any] = True,
                  mode: typing.Optional[typing.Any] = 'SET'):
    ''' Select keyframe points using circle selection

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param x: X
    :type x: typing.Optional[typing.Any]
    :param y: Y
    :type y: typing.Optional[typing.Any]
    :param radius: Radius
    :type radius: typing.Optional[typing.Any]
    :param wait_for_input: Wait for Input
    :type wait_for_input: typing.Union[bool, typing.Any]
    :param mode: Mode * ``SET`` Set, Set a new selection. * ``ADD`` Extend, Extend existing selection. * ``SUB`` Subtract, Subtract existing selection.
    :type mode: typing.Optional[typing.Any]
    '''

    pass


def select_column(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  mode: typing.Optional[typing.Any] = 'KEYS'):
    ''' Select all keyframes on the specified frame(s)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param mode: Mode
    :type mode: typing.Optional[typing.Any]
    '''

    pass


def select_lasso(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 path: typing.Optional[bpy.types.bpy_prop_collection[
                     'bpy.types.OperatorMousePath']] = None,
                 mode: typing.Optional[typing.Any] = 'SET'):
    ''' Select keyframe points using lasso selection

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param path: Path
    :type path: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorMousePath']]
    :param mode: Mode * ``SET`` Set, Set a new selection. * ``ADD`` Extend, Extend existing selection. * ``SUB`` Subtract, Subtract existing selection.
    :type mode: typing.Optional[typing.Any]
    '''

    pass


def select_leftright(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     mode: typing.Optional[typing.Any] = 'CHECK',
                     extend: typing.Union[bool, typing.Any] = False):
    ''' Select keyframes to the left or the right of the current frame

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param mode: Mode
    :type mode: typing.Optional[typing.Any]
    :param extend: Extend Select
    :type extend: typing.Union[bool, typing.Any]
    '''

    pass


def select_less(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None):
    ''' Deselect keyframes on ends of selection islands

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def select_linked(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None):
    ''' Select keyframes occurring in the same F-Curves as selected ones

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def select_more(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None):
    ''' Select keyframes beside already selected ones

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def smooth(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None):
    ''' Apply weighted moving means to make selected F-Curves less bumpy

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def snap(override_context: typing.Union[typing.
                                        Dict, 'bpy.types.Context'] = None,
         execution_context: typing.Union[str, int] = None,
         undo: typing.Optional[bool] = None,
         *,
         type: typing.Optional[typing.Any] = 'CFRA'):
    ''' Snap selected keyframes to the chosen times/values

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``CFRA`` Selection to Current Frame, Snap selected keyframes to the current frame. * ``VALUE`` Selection to Cursor Value, Set values of selected keyframes to the cursor value (Y/Horizontal component). * ``NEAREST_FRAME`` Selection to Nearest Frame, Snap selected keyframes to the nearest (whole) frame (use to fix accidental subframe offsets). * ``NEAREST_SECOND`` Selection to Nearest Second, Snap selected keyframes to the nearest second. * ``NEAREST_MARKER`` Selection to Nearest Marker, Snap selected keyframes to the nearest marker. * ``HORIZONTAL`` Flatten Handles, Flatten handles for a smoother transition.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def snap_cursor_value(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Place the cursor value on the average value of selected keyframes

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def sound_bake(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               filepath: typing.Union[str, typing.Any] = "",
               filter_blender: typing.Union[bool, typing.Any] = False,
               filter_backup: typing.Union[bool, typing.Any] = False,
               filter_image: typing.Union[bool, typing.Any] = False,
               filter_movie: typing.Union[bool, typing.Any] = True,
               filter_python: typing.Union[bool, typing.Any] = False,
               filter_font: typing.Union[bool, typing.Any] = False,
               filter_sound: typing.Union[bool, typing.Any] = True,
               filter_text: typing.Union[bool, typing.Any] = False,
               filter_archive: typing.Union[bool, typing.Any] = False,
               filter_btx: typing.Union[bool, typing.Any] = False,
               filter_collada: typing.Union[bool, typing.Any] = False,
               filter_alembic: typing.Union[bool, typing.Any] = False,
               filter_usd: typing.Union[bool, typing.Any] = False,
               filter_volume: typing.Union[bool, typing.Any] = False,
               filter_folder: typing.Union[bool, typing.Any] = True,
               filter_blenlib: typing.Union[bool, typing.Any] = False,
               filemode: typing.Optional[typing.Any] = 9,
               show_multiview: typing.Union[bool, typing.Any] = False,
               use_multiview: typing.Union[bool, typing.Any] = False,
               display_type: typing.Optional[typing.Any] = 'DEFAULT',
               sort_method: typing.Union[str, int, typing.Any] = '',
               low: typing.Optional[typing.Any] = 0.0,
               high: typing.Optional[typing.Any] = 100000.0,
               attack: typing.Optional[typing.Any] = 0.005,
               release: typing.Optional[typing.Any] = 0.2,
               threshold: typing.Optional[typing.Any] = 0.0,
               use_accumulate: typing.Union[bool, typing.Any] = False,
               use_additive: typing.Union[bool, typing.Any] = False,
               use_square: typing.Union[bool, typing.Any] = False,
               sthreshold: typing.Optional[typing.Any] = 0.1):
    ''' Bakes a sound wave to selected F-Curves

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param filter_blender: Filter .blend files
    :type filter_blender: typing.Union[bool, typing.Any]
    :param filter_backup: Filter .blend files
    :type filter_backup: typing.Union[bool, typing.Any]
    :param filter_image: Filter image files
    :type filter_image: typing.Union[bool, typing.Any]
    :param filter_movie: Filter movie files
    :type filter_movie: typing.Union[bool, typing.Any]
    :param filter_python: Filter python files
    :type filter_python: typing.Union[bool, typing.Any]
    :param filter_font: Filter font files
    :type filter_font: typing.Union[bool, typing.Any]
    :param filter_sound: Filter sound files
    :type filter_sound: typing.Union[bool, typing.Any]
    :param filter_text: Filter text files
    :type filter_text: typing.Union[bool, typing.Any]
    :param filter_archive: Filter archive files
    :type filter_archive: typing.Union[bool, typing.Any]
    :param filter_btx: Filter btx files
    :type filter_btx: typing.Union[bool, typing.Any]
    :param filter_collada: Filter COLLADA files
    :type filter_collada: typing.Union[bool, typing.Any]
    :param filter_alembic: Filter Alembic files
    :type filter_alembic: typing.Union[bool, typing.Any]
    :param filter_usd: Filter USD files
    :type filter_usd: typing.Union[bool, typing.Any]
    :param filter_volume: Filter OpenVDB volume files
    :type filter_volume: typing.Union[bool, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_blenlib: Filter Blender IDs
    :type filter_blenlib: typing.Union[bool, typing.Any]
    :param filemode: File Browser Mode, The setting for the file browser mode to load a .blend file, a library or a special file
    :type filemode: typing.Optional[typing.Any]
    :param show_multiview: Enable Multi-View
    :type show_multiview: typing.Union[bool, typing.Any]
    :param use_multiview: Use Multi-View
    :type use_multiview: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param low: Lowest Frequency, Cutoff frequency of a high-pass filter that is applied to the audio data
    :type low: typing.Optional[typing.Any]
    :param high: Highest Frequency, Cutoff frequency of a low-pass filter that is applied to the audio data
    :type high: typing.Optional[typing.Any]
    :param attack: Attack Time, Value for the hull curve calculation that tells how fast the hull curve can rise (the lower the value the steeper it can rise)
    :type attack: typing.Optional[typing.Any]
    :param release: Release Time, Value for the hull curve calculation that tells how fast the hull curve can fall (the lower the value the steeper it can fall)
    :type release: typing.Optional[typing.Any]
    :param threshold: Threshold, Minimum amplitude value needed to influence the hull curve
    :type threshold: typing.Optional[typing.Any]
    :param use_accumulate: Accumulate, Only the positive differences of the hull curve amplitudes are summarized to produce the output
    :type use_accumulate: typing.Union[bool, typing.Any]
    :param use_additive: Additive, The amplitudes of the hull curve are summarized (or, when Accumulate is enabled, both positive and negative differences are accumulated)
    :type use_additive: typing.Union[bool, typing.Any]
    :param use_square: Square, The output is a square curve (negative values always result in -1, and positive ones in 1)
    :type use_square: typing.Union[bool, typing.Any]
    :param sthreshold: Square Threshold, Square only: all values with an absolute amplitude lower than that result in 0
    :type sthreshold: typing.Optional[typing.Any]
    '''

    pass


def unbake(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None):
    ''' Un-Bake selected F-Points to F-Curves

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def view_all(override_context: typing.Union[typing.
                                            Dict, 'bpy.types.Context'] = None,
             execution_context: typing.Union[str, int] = None,
             undo: typing.Optional[bool] = None,
             *,
             include_handles: typing.Union[bool, typing.Any] = True):
    ''' Reset viewable area to show full keyframe range

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param include_handles: Include Handles, Include handles of keyframes when calculating extents
    :type include_handles: typing.Union[bool, typing.Any]
    '''

    pass


def view_frame(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None):
    ''' Move the view to the current frame

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def view_selected(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  include_handles: typing.Union[bool, typing.Any] = True):
    ''' Reset viewable area to show selected keyframe range

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param include_handles: Include Handles, Include handles of keyframes when calculating extents
    :type include_handles: typing.Union[bool, typing.Any]
    '''

    pass
