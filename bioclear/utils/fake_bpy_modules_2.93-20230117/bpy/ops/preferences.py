import sys
import typing
import bpy.types

GenericType = typing.TypeVar("GenericType")


def addon_disable(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  module: typing.Union[str, typing.Any] = ""):
    ''' Disable an add-on

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param module: Module, Module name of the add-on to disable
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def addon_enable(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 module: typing.Union[str, typing.Any] = ""):
    ''' Enable an add-on

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param module: Module, Module name of the add-on to enable
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def addon_expand(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 module: typing.Union[str, typing.Any] = ""):
    ''' Display information and preferences for this add-on

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param module: Module, Module name of the add-on to expand
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def addon_install(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  overwrite: typing.Union[bool, typing.Any] = True,
                  target: typing.Optional[typing.Any] = 'DEFAULT',
                  filepath: typing.Union[str, typing.Any] = "",
                  filter_folder: typing.Union[bool, typing.Any] = True,
                  filter_python: typing.Union[bool, typing.Any] = True,
                  filter_glob: typing.Union[str, typing.Any] = "*.py;*.zip"):
    ''' Install an add-on

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param overwrite: Overwrite, Remove existing add-ons with the same ID
    :type overwrite: typing.Union[bool, typing.Any]
    :param target: Target Path
    :type target: typing.Optional[typing.Any]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_python: Filter python
    :type filter_python: typing.Union[bool, typing.Any]
    :param filter_glob: filter_glob
    :type filter_glob: typing.Union[str, typing.Any]
    '''

    pass


def addon_refresh(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None):
    ''' Scan add-on directories for new modules :file: `startup/bl_operators/userpref.py\:560 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$560>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def addon_remove(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 module: typing.Union[str, typing.Any] = ""):
    ''' Delete the add-on from the file system

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param module: Module, Module name of the add-on to remove
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def addon_show(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               module: typing.Union[str, typing.Any] = ""):
    ''' Show add-on preferences

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param module: Module, Module name of the add-on to expand
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def app_template_install(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None,
                         *,
                         overwrite: typing.Union[bool, typing.Any] = True,
                         filepath: typing.Union[str, typing.Any] = "",
                         filter_folder: typing.Union[bool, typing.Any] = True,
                         filter_glob: typing.Union[str, typing.Any] = "*.zip"):
    ''' Install an application template

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param overwrite: Overwrite, Remove existing template with the same ID
    :type overwrite: typing.Union[bool, typing.Any]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_glob: filter_glob
    :type filter_glob: typing.Union[str, typing.Any]
    '''

    pass


def asset_library_add(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Add a path to a .blend file to be used by the Asset Browser as source of assets

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def asset_library_remove(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None,
                         *,
                         index: typing.Optional[typing.Any] = 0):
    ''' Remove a path to a .blend file, so the Asset Browser will not attempt to show it anymore

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param index: Index
    :type index: typing.Optional[typing.Any]
    '''

    pass


def autoexec_path_add(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Add path to exclude from auto-execution

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def autoexec_path_remove(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None,
                         *,
                         index: typing.Optional[typing.Any] = 0):
    ''' Remove path to exclude from auto-execution

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param index: Index
    :type index: typing.Optional[typing.Any]
    '''

    pass


def copy_prev(override_context: typing.Union[typing.
                                             Dict, 'bpy.types.Context'] = None,
              execution_context: typing.Union[str, int] = None,
              undo: typing.Optional[bool] = None):
    ''' Copy settings from previous version :file: `startup/bl_operators/userpref.py\:145 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$145>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def keyconfig_activate(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       filepath: typing.Union[str, typing.Any] = ""):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__.

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    '''

    pass


def keyconfig_export(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     all: typing.Union[bool, typing.Any] = False,
                     filepath: typing.Union[str, typing.Any] = "keymap.py",
                     filter_folder: typing.Union[bool, typing.Any] = True,
                     filter_text: typing.Union[bool, typing.Any] = True,
                     filter_python: typing.Union[bool, typing.Any] = True):
    ''' Export key configuration to a python script

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param all: All Keymaps, Write all keymaps (not just user modified)
    :type all: typing.Union[bool, typing.Any]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_text: Filter text
    :type filter_text: typing.Union[bool, typing.Any]
    :param filter_python: Filter python
    :type filter_python: typing.Union[bool, typing.Any]
    '''

    pass


def keyconfig_import(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     filepath: typing.Union[str, typing.Any] = "keymap.py",
                     filter_folder: typing.Union[bool, typing.Any] = True,
                     filter_text: typing.Union[bool, typing.Any] = True,
                     filter_python: typing.Union[bool, typing.Any] = True,
                     keep_original: typing.Union[bool, typing.Any] = True):
    ''' Import key configuration from a python script

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_text: Filter text
    :type filter_text: typing.Union[bool, typing.Any]
    :param filter_python: Filter python
    :type filter_python: typing.Union[bool, typing.Any]
    :param keep_original: Keep Original, Keep original file after copying to configuration folder
    :type keep_original: typing.Union[bool, typing.Any]
    '''

    pass


def keyconfig_remove(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None):
    ''' Remove key config :file: `startup/bl_operators/userpref.py\:408 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$408>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def keyconfig_test(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None):
    ''' Test key configuration for conflicts :file: `startup/bl_operators/userpref.py\:167 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$167>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def keyitem_add(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None):
    ''' Add key map item :file: `startup/bl_operators/userpref.py\:356 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$356>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def keyitem_remove(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   item_id: typing.Optional[typing.Any] = 0):
    ''' Remove key map item

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param item_id: Item Identifier, Identifier of the item to remove
    :type item_id: typing.Optional[typing.Any]
    '''

    pass


def keyitem_restore(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    item_id: typing.Optional[typing.Any] = 0):
    ''' Restore key map item

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param item_id: Item Identifier, Identifier of the item to restore
    :type item_id: typing.Optional[typing.Any]
    '''

    pass


def keymap_restore(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   all: typing.Union[bool, typing.Any] = False):
    ''' Restore key map(s)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param all: All Keymaps, Restore all keymaps to default
    :type all: typing.Union[bool, typing.Any]
    '''

    pass


def reset_default_theme(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None):
    ''' Reset to the default theme colors

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def studiolight_copy_settings(override_context: typing.
                              Union[typing.Dict, 'bpy.types.Context'] = None,
                              execution_context: typing.Union[str, int] = None,
                              undo: typing.Optional[bool] = None,
                              *,
                              index: typing.Optional[typing.Any] = 0):
    ''' Copy Studio Light settings to the Studio Light editor

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param index: index
    :type index: typing.Optional[typing.Any]
    '''

    pass


def studiolight_install(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        files: typing.Optional[bpy.types.bpy_prop_collection[
            'bpy.types.OperatorFileListElement']] = None,
        directory: typing.Union[str, typing.Any] = "",
        filter_folder: typing.Union[bool, typing.Any] = True,
        filter_glob: typing.Union[str, typing.Any] = "*.png;*.jpg;*.hdr;*.exr",
        type: typing.Optional[typing.Any] = 'MATCAP'):
    ''' Install a user defined light

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param files: File Path
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
    :param directory: directory
    :type directory: typing.Union[str, typing.Any]
    :param filter_folder: Filter Folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_glob: filter_glob
    :type filter_glob: typing.Union[str, typing.Any]
    :param type: Type * ``MATCAP`` MatCap, Install custom MatCaps. * ``WORLD`` World, Install custom HDRIs. * ``STUDIO`` Studio, Install custom Studio Lights.
    :type type: typing.Optional[typing.Any]
    '''

    pass


def studiolight_new(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    filename: typing.Union[str, typing.Any] = "StudioLight"):
    ''' Save custom studio light from the studio light editor settings

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filename: Name
    :type filename: typing.Union[str, typing.Any]
    '''

    pass


def studiolight_show(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None):
    ''' Show light preferences :file: `startup/bl_operators/userpref.py\:1117 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/userpref.py$1117>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def studiolight_uninstall(override_context: typing.
                          Union[typing.Dict, 'bpy.types.Context'] = None,
                          execution_context: typing.Union[str, int] = None,
                          undo: typing.Optional[bool] = None,
                          *,
                          index: typing.Optional[typing.Any] = 0):
    ''' Delete Studio Light

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param index: index
    :type index: typing.Optional[typing.Any]
    '''

    pass


def theme_install(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  overwrite: typing.Union[bool, typing.Any] = True,
                  filepath: typing.Union[str, typing.Any] = "",
                  filter_folder: typing.Union[bool, typing.Any] = True,
                  filter_glob: typing.Union[str, typing.Any] = "*.xml"):
    ''' Load and apply a Blender XML theme file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param overwrite: Overwrite, Remove existing theme file if exists
    :type overwrite: typing.Union[bool, typing.Any]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    :param filter_folder: Filter folders
    :type filter_folder: typing.Union[bool, typing.Any]
    :param filter_glob: filter_glob
    :type filter_glob: typing.Union[str, typing.Any]
    '''

    pass
