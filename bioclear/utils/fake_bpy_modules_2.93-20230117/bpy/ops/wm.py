import sys
import typing
import bpy.types
import bl_operators.wm

GenericType = typing.TypeVar("GenericType")


def alembic_export(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        filepath: typing.Union[str, typing.Any] = "",
        check_existing: typing.Union[bool, typing.Any] = True,
        filter_blender: typing.Union[bool, typing.Any] = False,
        filter_backup: typing.Union[bool, typing.Any] = False,
        filter_image: typing.Union[bool, typing.Any] = False,
        filter_movie: typing.Union[bool, typing.Any] = False,
        filter_python: typing.Union[bool, typing.Any] = False,
        filter_font: typing.Union[bool, typing.Any] = False,
        filter_sound: typing.Union[bool, typing.Any] = False,
        filter_text: typing.Union[bool, typing.Any] = False,
        filter_archive: typing.Union[bool, typing.Any] = False,
        filter_btx: typing.Union[bool, typing.Any] = False,
        filter_collada: typing.Union[bool, typing.Any] = False,
        filter_alembic: typing.Union[bool, typing.Any] = True,
        filter_usd: typing.Union[bool, typing.Any] = False,
        filter_volume: typing.Union[bool, typing.Any] = False,
        filter_folder: typing.Union[bool, typing.Any] = True,
        filter_blenlib: typing.Union[bool, typing.Any] = False,
        filemode: typing.Optional[typing.Any] = 8,
        display_type: typing.Optional[typing.Any] = 'DEFAULT',
        sort_method: typing.Union[str, int, typing.Any] = '',
        start: typing.Optional[typing.Any] = -2147483648,
        end: typing.Optional[typing.Any] = -2147483648,
        xsamples: typing.Optional[typing.Any] = 1,
        gsamples: typing.Optional[typing.Any] = 1,
        sh_open: typing.Optional[typing.Any] = 0.0,
        sh_close: typing.Optional[typing.Any] = 1.0,
        selected: typing.Union[bool, typing.Any] = False,
        renderable_only: typing.Union[bool, typing.Any] = True,
        visible_objects_only: typing.Union[bool, typing.Any] = False,
        flatten: typing.Union[bool, typing.Any] = False,
        uvs: typing.Union[bool, typing.Any] = True,
        packuv: typing.Union[bool, typing.Any] = True,
        normals: typing.Union[bool, typing.Any] = True,
        vcolors: typing.Union[bool, typing.Any] = False,
        face_sets: typing.Union[bool, typing.Any] = False,
        subdiv_schema: typing.Union[bool, typing.Any] = False,
        apply_subdiv: typing.Union[bool, typing.Any] = False,
        curves_as_mesh: typing.Union[bool, typing.Any] = False,
        use_instancing: typing.Union[bool, typing.Any] = True,
        global_scale: typing.Optional[typing.Any] = 1.0,
        triangulate: typing.Union[bool, typing.Any] = False,
        quad_method: typing.Optional[typing.Any] = 'SHORTEST_DIAGONAL',
        ngon_method: typing.Optional[typing.Any] = 'BEAUTY',
        export_hair: typing.Union[bool, typing.Any] = True,
        export_particles: typing.Union[bool, typing.Any] = True,
        export_custom_properties: typing.Union[bool, typing.Any] = True,
        as_background_job: typing.Union[bool, typing.Any] = False,
        init_scene_frame_range: typing.Union[bool, typing.Any] = False):
    ''' Export current scene in an Alembic archive

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param start: Start Frame, Start frame of the export, use the default value to take the start frame of the current scene
    :type start: typing.Optional[typing.Any]
    :param end: End Frame, End frame of the export, use the default value to take the end frame of the current scene
    :type end: typing.Optional[typing.Any]
    :param xsamples: Transform Samples, Number of times per frame transformations are sampled
    :type xsamples: typing.Optional[typing.Any]
    :param gsamples: Geometry Samples, Number of times per frame object data are sampled
    :type gsamples: typing.Optional[typing.Any]
    :param sh_open: Shutter Open, Time at which the shutter is open
    :type sh_open: typing.Optional[typing.Any]
    :param sh_close: Shutter Close, Time at which the shutter is closed
    :type sh_close: typing.Optional[typing.Any]
    :param selected: Selected Objects Only, Export only selected objects
    :type selected: typing.Union[bool, typing.Any]
    :param renderable_only: Renderable Objects Only, Export only objects marked renderable in the outliner
    :type renderable_only: typing.Union[bool, typing.Any]
    :param visible_objects_only: Visible Objects Only, Export only objects that are visible
    :type visible_objects_only: typing.Union[bool, typing.Any]
    :param flatten: Flatten Hierarchy, Do not preserve objects' parent/children relationship
    :type flatten: typing.Union[bool, typing.Any]
    :param uvs: UVs, Export UVs
    :type uvs: typing.Union[bool, typing.Any]
    :param packuv: Pack UV Islands, Export UVs with packed island
    :type packuv: typing.Union[bool, typing.Any]
    :param normals: Normals, Export normals
    :type normals: typing.Union[bool, typing.Any]
    :param vcolors: Vertex Colors, Export vertex colors
    :type vcolors: typing.Union[bool, typing.Any]
    :param face_sets: Face Sets, Export per face shading group assignments
    :type face_sets: typing.Union[bool, typing.Any]
    :param subdiv_schema: Use Subdivision Schema, Export meshes using Alembic's subdivision schema
    :type subdiv_schema: typing.Union[bool, typing.Any]
    :param apply_subdiv: Apply Subdivision Surface, Export subdivision surfaces as meshes
    :type apply_subdiv: typing.Union[bool, typing.Any]
    :param curves_as_mesh: Curves as Mesh, Export curves and NURBS surfaces as meshes
    :type curves_as_mesh: typing.Union[bool, typing.Any]
    :param use_instancing: Use Instancing, Export data of duplicated objects as Alembic instances; speeds up the export and can be disabled for compatibility with other software
    :type use_instancing: typing.Union[bool, typing.Any]
    :param global_scale: Scale, Value by which to enlarge or shrink the objects with respect to the world's origin
    :type global_scale: typing.Optional[typing.Any]
    :param triangulate: Triangulate, Export polygons (quads and n-gons) as triangles
    :type triangulate: typing.Union[bool, typing.Any]
    :param quad_method: Quad Method, Method for splitting the quads into triangles * ``BEAUTY`` Beauty, Split the quads in nice triangles, slower method. * ``FIXED`` Fixed, Split the quads on the first and third vertices. * ``FIXED_ALTERNATE`` Fixed Alternate, Split the quads on the 2nd and 4th vertices. * ``SHORTEST_DIAGONAL`` Shortest Diagonal, Split the quads based on the distance between the vertices.
    :type quad_method: typing.Optional[typing.Any]
    :param ngon_method: N-gon Method, Method for splitting the n-gons into triangles * ``BEAUTY`` Beauty, Arrange the new triangles evenly (slow). * ``CLIP`` Clip, Split the polygons with an ear clipping algorithm.
    :type ngon_method: typing.Optional[typing.Any]
    :param export_hair: Export Hair, Exports hair particle systems as animated curves
    :type export_hair: typing.Union[bool, typing.Any]
    :param export_particles: Export Particles, Exports non-hair particle systems
    :type export_particles: typing.Union[bool, typing.Any]
    :param export_custom_properties: Export Custom Properties, Export custom properties to Alembic .userProperties
    :type export_custom_properties: typing.Union[bool, typing.Any]
    :param as_background_job: Run as Background Job, Enable this to run the import in the background, disable to block Blender while importing. This option is deprecated; EXECUTE this operator to run in the foreground, and INVOKE it to run as a background job
    :type as_background_job: typing.Union[bool, typing.Any]
    :type init_scene_frame_range: typing.Union[bool, typing.Any]
    '''

    pass


def alembic_import(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   filepath: typing.Union[str, typing.Any] = "",
                   filter_blender: typing.Union[bool, typing.Any] = False,
                   filter_backup: typing.Union[bool, typing.Any] = False,
                   filter_image: typing.Union[bool, typing.Any] = False,
                   filter_movie: typing.Union[bool, typing.Any] = False,
                   filter_python: typing.Union[bool, typing.Any] = False,
                   filter_font: typing.Union[bool, typing.Any] = False,
                   filter_sound: typing.Union[bool, typing.Any] = False,
                   filter_text: typing.Union[bool, typing.Any] = False,
                   filter_archive: typing.Union[bool, typing.Any] = False,
                   filter_btx: typing.Union[bool, typing.Any] = False,
                   filter_collada: typing.Union[bool, typing.Any] = False,
                   filter_alembic: typing.Union[bool, typing.Any] = True,
                   filter_usd: typing.Union[bool, typing.Any] = False,
                   filter_volume: typing.Union[bool, typing.Any] = False,
                   filter_folder: typing.Union[bool, typing.Any] = True,
                   filter_blenlib: typing.Union[bool, typing.Any] = False,
                   filemode: typing.Optional[typing.Any] = 8,
                   relative_path: typing.Union[bool, typing.Any] = True,
                   display_type: typing.Optional[typing.Any] = 'DEFAULT',
                   sort_method: typing.Union[str, int, typing.Any] = '',
                   scale: typing.Optional[typing.Any] = 1.0,
                   set_frame_range: typing.Union[bool, typing.Any] = True,
                   validate_meshes: typing.Union[bool, typing.Any] = False,
                   is_sequence: typing.Union[bool, typing.Any] = False,
                   as_background_job: typing.Union[bool, typing.Any] = False):
    ''' Load an Alembic archive

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
    :param relative_path: Relative Path, Select the file relative to the blend file
    :type relative_path: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param scale: Scale, Value by which to enlarge or shrink the objects with respect to the world's origin
    :type scale: typing.Optional[typing.Any]
    :param set_frame_range: Set Frame Range, If checked, update scene's start and end frame to match those of the Alembic archive
    :type set_frame_range: typing.Union[bool, typing.Any]
    :param validate_meshes: Validate Meshes, Check imported mesh objects for invalid data (slow)
    :type validate_meshes: typing.Union[bool, typing.Any]
    :param is_sequence: Is Sequence, Set to true if the cache is split into separate files
    :type is_sequence: typing.Union[bool, typing.Any]
    :param as_background_job: Run as Background Job, Enable this to run the export in the background, disable to block Blender while exporting. This option is deprecated; EXECUTE this operator to run in the foreground, and INVOKE it to run as a background job
    :type as_background_job: typing.Union[bool, typing.Any]
    '''

    pass


def append(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None,
           *,
           filepath: typing.Union[str, typing.Any] = "",
           directory: typing.Union[str, typing.Any] = "",
           filename: typing.Union[str, typing.Any] = "",
           files: typing.Optional[bpy.types.bpy_prop_collection[
               'bpy.types.OperatorFileListElement']] = None,
           filter_blender: typing.Union[bool, typing.Any] = True,
           filter_backup: typing.Union[bool, typing.Any] = False,
           filter_image: typing.Union[bool, typing.Any] = False,
           filter_movie: typing.Union[bool, typing.Any] = False,
           filter_python: typing.Union[bool, typing.Any] = False,
           filter_font: typing.Union[bool, typing.Any] = False,
           filter_sound: typing.Union[bool, typing.Any] = False,
           filter_text: typing.Union[bool, typing.Any] = False,
           filter_archive: typing.Union[bool, typing.Any] = False,
           filter_btx: typing.Union[bool, typing.Any] = False,
           filter_collada: typing.Union[bool, typing.Any] = False,
           filter_alembic: typing.Union[bool, typing.Any] = False,
           filter_usd: typing.Union[bool, typing.Any] = False,
           filter_volume: typing.Union[bool, typing.Any] = False,
           filter_folder: typing.Union[bool, typing.Any] = True,
           filter_blenlib: typing.Union[bool, typing.Any] = True,
           filemode: typing.Optional[typing.Any] = 1,
           display_type: typing.Optional[typing.Any] = 'DEFAULT',
           sort_method: typing.Union[str, int, typing.Any] = '',
           link: typing.Union[bool, typing.Any] = False,
           autoselect: typing.Union[bool, typing.Any] = True,
           active_collection: typing.Union[bool, typing.Any] = True,
           instance_collections: typing.Union[bool, typing.Any] = False,
           instance_object_data: typing.Union[bool, typing.Any] = True,
           set_fake: typing.Union[bool, typing.Any] = False,
           use_recursive: typing.Union[bool, typing.Any] = True):
    ''' Append from a Library .blend file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param directory: Directory, Directory of the file
    :type directory: typing.Union[str, typing.Any]
    :param filename: File Name, Name of the file
    :type filename: typing.Union[str, typing.Any]
    :param files: Files
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param link: Link, Link the objects or data-blocks rather than appending
    :type link: typing.Union[bool, typing.Any]
    :param autoselect: Select, Select new objects
    :type autoselect: typing.Union[bool, typing.Any]
    :param active_collection: Active Collection, Put new objects on the active collection
    :type active_collection: typing.Union[bool, typing.Any]
    :param instance_collections: Instance Collections, Create instances for collections, rather than adding them directly to the scene
    :type instance_collections: typing.Union[bool, typing.Any]
    :param instance_object_data: Instance Object Data, Create instances for object data which are not referenced by any objects
    :type instance_object_data: typing.Union[bool, typing.Any]
    :param set_fake: Fake User, Set "Fake User" for appended items (except objects and collections)
    :type set_fake: typing.Union[bool, typing.Any]
    :param use_recursive: Localize All, Localize all appended data, including those indirectly linked from other libraries
    :type use_recursive: typing.Union[bool, typing.Any]
    '''

    pass


def batch_rename(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 data_type: typing.Optional[typing.Any] = 'OBJECT',
                 data_source: typing.Optional[typing.Any] = 'SELECT',
                 actions: typing.Optional[bpy.types.bpy_prop_collection[
                     'bl_operators.wm.BatchRenameAction']] = None):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__.

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_type: Type, Type of data to rename
    :type data_type: typing.Optional[typing.Any]
    :param data_source: Source
    :type data_source: typing.Optional[typing.Any]
    :param actions: actions
    :type actions: typing.Optional[bpy.types.bpy_prop_collection['bl_operators.wm.BatchRenameAction']]
    '''

    pass


def blend_strings_utf8_validate(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None):
    ''' Check and fix all strings in current .blend file to be valid UTF-8 Unicode (needed for some old, 2.4x area files) :file: `startup/bl_operators/file.py\:297 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/file.py$297>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def call_menu(override_context: typing.Union[typing.
                                             Dict, 'bpy.types.Context'] = None,
              execution_context: typing.Union[str, int] = None,
              undo: typing.Optional[bool] = None,
              *,
              name: typing.Union[str, typing.Any] = ""):
    ''' Open a predefined menu

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the menu
    :type name: typing.Union[str, typing.Any]
    '''

    pass


def call_menu_pie(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  name: typing.Union[str, typing.Any] = ""):
    ''' Open a predefined pie menu

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the pie menu
    :type name: typing.Union[str, typing.Any]
    '''

    pass


def call_panel(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               name: typing.Union[str, typing.Any] = "",
               keep_open: typing.Union[bool, typing.Any] = True):
    ''' Open a predefined panel

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the menu
    :type name: typing.Union[str, typing.Any]
    :param keep_open: Keep Open
    :type keep_open: typing.Union[bool, typing.Any]
    '''

    pass


def collada_export(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        filepath: typing.Union[str, typing.Any] = "",
        check_existing: typing.Union[bool, typing.Any] = True,
        filter_blender: typing.Union[bool, typing.Any] = False,
        filter_backup: typing.Union[bool, typing.Any] = False,
        filter_image: typing.Union[bool, typing.Any] = False,
        filter_movie: typing.Union[bool, typing.Any] = False,
        filter_python: typing.Union[bool, typing.Any] = False,
        filter_font: typing.Union[bool, typing.Any] = False,
        filter_sound: typing.Union[bool, typing.Any] = False,
        filter_text: typing.Union[bool, typing.Any] = False,
        filter_archive: typing.Union[bool, typing.Any] = False,
        filter_btx: typing.Union[bool, typing.Any] = False,
        filter_collada: typing.Union[bool, typing.Any] = True,
        filter_alembic: typing.Union[bool, typing.Any] = False,
        filter_usd: typing.Union[bool, typing.Any] = False,
        filter_volume: typing.Union[bool, typing.Any] = False,
        filter_folder: typing.Union[bool, typing.Any] = True,
        filter_blenlib: typing.Union[bool, typing.Any] = False,
        filemode: typing.Optional[typing.Any] = 8,
        display_type: typing.Optional[typing.Any] = 'DEFAULT',
        sort_method: typing.Union[str, int, typing.Any] = '',
        prop_bc_export_ui_section: typing.Optional[typing.Any] = 'main',
        apply_modifiers: typing.Union[bool, typing.Any] = False,
        export_mesh_type: typing.Optional[typing.Any] = 0,
        export_mesh_type_selection: typing.Optional[typing.Any] = 'view',
        export_global_forward_selection: typing.Optional[typing.Any] = 'Y',
        export_global_up_selection: typing.Optional[typing.Any] = 'Z',
        apply_global_orientation: typing.Union[bool, typing.Any] = False,
        selected: typing.Union[bool, typing.Any] = False,
        include_children: typing.Union[bool, typing.Any] = False,
        include_armatures: typing.Union[bool, typing.Any] = False,
        include_shapekeys: typing.Union[bool, typing.Any] = False,
        deform_bones_only: typing.Union[bool, typing.Any] = False,
        include_animations: typing.Union[bool, typing.Any] = True,
        include_all_actions: typing.Union[bool, typing.Any] = True,
        export_animation_type_selection: typing.Optional[typing.
                                                         Any] = 'sample',
        sampling_rate: typing.Optional[typing.Any] = 1,
        keep_smooth_curves: typing.Union[bool, typing.Any] = False,
        keep_keyframes: typing.Union[bool, typing.Any] = False,
        keep_flat_curves: typing.Union[bool, typing.Any] = False,
        active_uv_only: typing.Union[bool, typing.Any] = False,
        use_texture_copies: typing.Union[bool, typing.Any] = True,
        triangulate: typing.Union[bool, typing.Any] = True,
        use_object_instantiation: typing.Union[bool, typing.Any] = True,
        use_blender_profile: typing.Union[bool, typing.Any] = True,
        sort_by_name: typing.Union[bool, typing.Any] = False,
        export_object_transformation_type: typing.Optional[typing.Any] = 0,
        export_object_transformation_type_selection: typing.Optional[
            typing.Any] = 'matrix',
        export_animation_transformation_type: typing.Optional[typing.Any] = 0,
        export_animation_transformation_type_selection: typing.Optional[
            typing.Any] = 'matrix',
        open_sim: typing.Union[bool, typing.Any] = False,
        limit_precision: typing.Union[bool, typing.Any] = False,
        keep_bind_info: typing.Union[bool, typing.Any] = False):
    ''' Save a Collada file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param prop_bc_export_ui_section: Export Section, Only for User Interface organization * ``main`` Main, Data export section. * ``geometry`` Geom, Geometry export section. * ``armature`` Arm, Armature export section. * ``animation`` Anim, Animation export section. * ``collada`` Extra, Collada export section.
    :type prop_bc_export_ui_section: typing.Optional[typing.Any]
    :param apply_modifiers: Apply Modifiers, Apply modifiers to exported mesh (non destructive))
    :type apply_modifiers: typing.Union[bool, typing.Any]
    :param export_mesh_type: Resolution, Modifier resolution for export
    :type export_mesh_type: typing.Optional[typing.Any]
    :param export_mesh_type_selection: Resolution, Modifier resolution for export * ``view`` Viewport, Apply modifier's viewport settings. * ``render`` Render, Apply modifier's render settings.
    :type export_mesh_type_selection: typing.Optional[typing.Any]
    :param export_global_forward_selection: Global Forward Axis, Global Forward axis for export * ``X`` X, Global Forward is positive X Axis. * ``Y`` Y, Global Forward is positive Y Axis. * ``Z`` Z, Global Forward is positive Z Axis. * ``-X`` -X, Global Forward is negative X Axis. * ``-Y`` -Y, Global Forward is negative Y Axis. * ``-Z`` -Z, Global Forward is negative Z Axis.
    :type export_global_forward_selection: typing.Optional[typing.Any]
    :param export_global_up_selection: Global Up Axis, Global Up axis for export * ``X`` X, Global UP is positive X Axis. * ``Y`` Y, Global UP is positive Y Axis. * ``Z`` Z, Global UP is positive Z Axis. * ``-X`` -X, Global UP is negative X Axis. * ``-Y`` -Y, Global UP is negative Y Axis. * ``-Z`` -Z, Global UP is negative Z Axis.
    :type export_global_up_selection: typing.Optional[typing.Any]
    :param apply_global_orientation: Apply Global Orientation, Rotate all root objects to match the global orientation settings otherwise set the global orientation per Collada asset
    :type apply_global_orientation: typing.Union[bool, typing.Any]
    :param selected: Selection Only, Export only selected elements
    :type selected: typing.Union[bool, typing.Any]
    :param include_children: Include Children, Export all children of selected objects (even if not selected)
    :type include_children: typing.Union[bool, typing.Any]
    :param include_armatures: Include Armatures, Export related armatures (even if not selected)
    :type include_armatures: typing.Union[bool, typing.Any]
    :param include_shapekeys: Include Shape Keys, Export all Shape Keys from Mesh Objects
    :type include_shapekeys: typing.Union[bool, typing.Any]
    :param deform_bones_only: Deform Bones Only, Only export deforming bones with armatures
    :type deform_bones_only: typing.Union[bool, typing.Any]
    :param include_animations: Include Animations, Export animations if available (exporting animations will enforce the decomposition of node transforms into <translation> <rotation> and <scale> components)
    :type include_animations: typing.Union[bool, typing.Any]
    :param include_all_actions: Include all Actions, Export also unassigned actions (this allows you to export entire animation libraries for your character(s))
    :type include_all_actions: typing.Union[bool, typing.Any]
    :param export_animation_type_selection: Key Type, Type for exported animations (use sample keys or Curve keys) * ``sample`` Samples, Export Sampled points guided by sampling rate. * ``keys`` Curves, Export Curves (note: guided by curve keys).
    :type export_animation_type_selection: typing.Optional[typing.Any]
    :param sampling_rate: Sampling Rate, The distance between 2 keyframes (1 to key every frame)
    :type sampling_rate: typing.Optional[typing.Any]
    :param keep_smooth_curves: Keep Smooth curves, Export also the curve handles (if available) (this does only work when the inverse parent matrix is the unity matrix, otherwise you may end up with odd results)
    :type keep_smooth_curves: typing.Union[bool, typing.Any]
    :param keep_keyframes: Keep Keyframes, Use existing keyframes as additional sample points (this helps when you want to keep manual tweaks)
    :type keep_keyframes: typing.Union[bool, typing.Any]
    :param keep_flat_curves: All Keyed Curves, Export also curves which have only one key or are totally flat
    :type keep_flat_curves: typing.Union[bool, typing.Any]
    :param active_uv_only: Only Selected UV Map, Export only the selected UV Map
    :type active_uv_only: typing.Union[bool, typing.Any]
    :param use_texture_copies: Copy, Copy textures to same folder where the .dae file is exported
    :type use_texture_copies: typing.Union[bool, typing.Any]
    :param triangulate: Triangulate, Export polygons (quads and n-gons) as triangles
    :type triangulate: typing.Union[bool, typing.Any]
    :param use_object_instantiation: Use Object Instances, Instantiate multiple Objects from same Data
    :type use_object_instantiation: typing.Union[bool, typing.Any]
    :param use_blender_profile: Use Blender Profile, Export additional Blender specific information (for material, shaders, bones, etc.)
    :type use_blender_profile: typing.Union[bool, typing.Any]
    :param sort_by_name: Sort by Object name, Sort exported data by Object name
    :type sort_by_name: typing.Union[bool, typing.Any]
    :param export_object_transformation_type: Transform, Object Transformation type for translation, scale and rotation
    :type export_object_transformation_type: typing.Optional[typing.Any]
    :param export_object_transformation_type_selection: Transform, Object Transformation type for translation, scale and rotation * ``matrix`` Matrix, Use <matrix> representation for exported transformations. * ``decomposed`` Decomposed, Use <rotate>, <translate> and <scale> representation for exported transformations.
    :type export_object_transformation_type_selection: typing.Optional[typing.Any]
    :param export_animation_transformation_type: Transform, Transformation type for translation, scale and rotation. Note: The Animation transformation type in the Anim Tab is always equal to the Object transformation type in the Geom tab
    :type export_animation_transformation_type: typing.Optional[typing.Any]
    :param export_animation_transformation_type_selection: Transform, Transformation type for translation, scale and rotation. Note: The Animation transformation type in the Anim Tab is always equal to the Object transformation type in the Geom tab * ``matrix`` Matrix, Use <matrix> representation for exported transformations. * ``decomposed`` Decomposed, Use <rotate>, <translate> and <scale> representation for exported transformations.
    :type export_animation_transformation_type_selection: typing.Optional[typing.Any]
    :param open_sim: Export to SL/OpenSim, Compatibility mode for SL, OpenSim and other compatible online worlds
    :type open_sim: typing.Union[bool, typing.Any]
    :param limit_precision: Limit Precision, Reduce the precision of the exported data to 6 digits
    :type limit_precision: typing.Union[bool, typing.Any]
    :param keep_bind_info: Keep Bind Info, Store Bindpose information in custom bone properties for later use during Collada export
    :type keep_bind_info: typing.Union[bool, typing.Any]
    '''

    pass


def collada_import(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   filepath: typing.Union[str, typing.Any] = "",
                   filter_blender: typing.Union[bool, typing.Any] = False,
                   filter_backup: typing.Union[bool, typing.Any] = False,
                   filter_image: typing.Union[bool, typing.Any] = False,
                   filter_movie: typing.Union[bool, typing.Any] = False,
                   filter_python: typing.Union[bool, typing.Any] = False,
                   filter_font: typing.Union[bool, typing.Any] = False,
                   filter_sound: typing.Union[bool, typing.Any] = False,
                   filter_text: typing.Union[bool, typing.Any] = False,
                   filter_archive: typing.Union[bool, typing.Any] = False,
                   filter_btx: typing.Union[bool, typing.Any] = False,
                   filter_collada: typing.Union[bool, typing.Any] = True,
                   filter_alembic: typing.Union[bool, typing.Any] = False,
                   filter_usd: typing.Union[bool, typing.Any] = False,
                   filter_volume: typing.Union[bool, typing.Any] = False,
                   filter_folder: typing.Union[bool, typing.Any] = True,
                   filter_blenlib: typing.Union[bool, typing.Any] = False,
                   filemode: typing.Optional[typing.Any] = 8,
                   display_type: typing.Optional[typing.Any] = 'DEFAULT',
                   sort_method: typing.Union[str, int, typing.Any] = '',
                   import_units: typing.Union[bool, typing.Any] = False,
                   fix_orientation: typing.Union[bool, typing.Any] = False,
                   find_chains: typing.Union[bool, typing.Any] = False,
                   auto_connect: typing.Union[bool, typing.Any] = False,
                   min_chain_length: typing.Optional[typing.Any] = 0,
                   keep_bind_info: typing.Union[bool, typing.Any] = False):
    ''' Load a Collada file

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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param import_units: Import Units, If disabled match import to Blender's current Unit settings, otherwise use the settings from the Imported scene
    :type import_units: typing.Union[bool, typing.Any]
    :param fix_orientation: Fix Leaf Bones, Fix Orientation of Leaf Bones (Collada does only support Joints)
    :type fix_orientation: typing.Union[bool, typing.Any]
    :param find_chains: Find Bone Chains, Find best matching Bone Chains and ensure bones in chain are connected
    :type find_chains: typing.Union[bool, typing.Any]
    :param auto_connect: Auto Connect, Set use_connect for parent bones which have exactly one child bone
    :type auto_connect: typing.Union[bool, typing.Any]
    :param min_chain_length: Minimum Chain Length, When searching Bone Chains disregard chains of length below this value
    :type min_chain_length: typing.Optional[typing.Any]
    :param keep_bind_info: Keep Bind Info, Store Bindpose information in custom bone properties for later use during Collada export
    :type keep_bind_info: typing.Union[bool, typing.Any]
    '''

    pass


def context_collection_boolean_set(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        data_path_iter: typing.Union[str, typing.Any] = "",
        data_path_item: typing.Union[str, typing.Any] = "",
        type: typing.Optional[typing.Any] = 'TOGGLE'):
    ''' Set boolean values for a collection of items

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path_iter: data_path_iter, The data path relative to the context, must point to an iterable
    :type data_path_iter: typing.Union[str, typing.Any]
    :param data_path_item: data_path_item, The data path from each iterable to the value (int or float)
    :type data_path_item: typing.Union[str, typing.Any]
    :param type: Type
    :type type: typing.Optional[typing.Any]
    '''

    pass


def context_cycle_array(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        data_path: typing.Union[str, typing.Any] = "",
                        reverse: typing.Union[bool, typing.Any] = False):
    ''' Set a context array value (useful for cycling the active mesh edit mode)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param reverse: Reverse, Cycle backwards
    :type reverse: typing.Union[bool, typing.Any]
    '''

    pass


def context_cycle_enum(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       data_path: typing.Union[str, typing.Any] = "",
                       reverse: typing.Union[bool, typing.Any] = False,
                       wrap: typing.Union[bool, typing.Any] = False):
    ''' Toggle a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param reverse: Reverse, Cycle backwards
    :type reverse: typing.Union[bool, typing.Any]
    :param wrap: Wrap, Wrap back to the first/last values
    :type wrap: typing.Union[bool, typing.Any]
    '''

    pass


def context_cycle_int(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      reverse: typing.Union[bool, typing.Any] = False,
                      wrap: typing.Union[bool, typing.Any] = False):
    ''' Set a context value (useful for cycling active material, vertex keys, groups, etc.)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param reverse: Reverse, Cycle backwards
    :type reverse: typing.Union[bool, typing.Any]
    :param wrap: Wrap, Wrap back to the first/last values
    :type wrap: typing.Union[bool, typing.Any]
    '''

    pass


def context_menu_enum(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = ""):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__.

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    '''

    pass


def context_modal_mouse(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        data_path_iter: typing.Union[str, typing.Any] = "",
                        data_path_item: typing.Union[str, typing.Any] = "",
                        header_text: typing.Union[str, typing.Any] = "",
                        input_scale: typing.Optional[typing.Any] = 0.01,
                        invert: typing.Union[bool, typing.Any] = False,
                        initial_x: typing.Optional[typing.Any] = 0):
    ''' Adjust arbitrary values with mouse input

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path_iter: data_path_iter, The data path relative to the context, must point to an iterable
    :type data_path_iter: typing.Union[str, typing.Any]
    :param data_path_item: data_path_item, The data path from each iterable to the value (int or float)
    :type data_path_item: typing.Union[str, typing.Any]
    :param header_text: Header Text, Text to display in header during scale
    :type header_text: typing.Union[str, typing.Any]
    :param input_scale: input_scale, Scale the mouse movement by this value before applying the delta
    :type input_scale: typing.Optional[typing.Any]
    :param invert: invert, Invert the mouse input
    :type invert: typing.Union[bool, typing.Any]
    :param initial_x: initial_x
    :type initial_x: typing.Optional[typing.Any]
    '''

    pass


def context_pie_enum(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     data_path: typing.Union[str, typing.Any] = ""):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__.

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    '''

    pass


def context_scale_float(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        data_path: typing.Union[str, typing.Any] = "",
                        value: typing.Optional[typing.Any] = 1.0):
    ''' Scale a float context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assign value
    :type value: typing.Optional[typing.Any]
    '''

    pass


def context_scale_int(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      value: typing.Optional[typing.Any] = 1.0,
                      always_step: typing.Union[bool, typing.Any] = True):
    ''' Scale an int context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assign value
    :type value: typing.Optional[typing.Any]
    :param always_step: Always Step, Always adjust the value by a minimum of 1 when 'value' is not 1.0
    :type always_step: typing.Union[bool, typing.Any]
    '''

    pass


def context_set_boolean(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        data_path: typing.Union[str, typing.Any] = "",
                        value: typing.Union[bool, typing.Any] = True):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assignment value
    :type value: typing.Union[bool, typing.Any]
    '''

    pass


def context_set_enum(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     data_path: typing.Union[str, typing.Any] = "",
                     value: typing.Union[str, typing.Any] = ""):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assignment value (as a string)
    :type value: typing.Union[str, typing.Any]
    '''

    pass


def context_set_float(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      value: typing.Optional[typing.Any] = 0.0,
                      relative: typing.Union[bool, typing.Any] = False):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assignment value
    :type value: typing.Optional[typing.Any]
    :param relative: Relative, Apply relative to the current value (delta)
    :type relative: typing.Union[bool, typing.Any]
    '''

    pass


def context_set_id(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   data_path: typing.Union[str, typing.Any] = "",
                   value: typing.Union[str, typing.Any] = ""):
    ''' Set a context value to an ID data-block

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assign value
    :type value: typing.Union[str, typing.Any]
    '''

    pass


def context_set_int(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    data_path: typing.Union[str, typing.Any] = "",
                    value: typing.Optional[typing.Any] = 0,
                    relative: typing.Union[bool, typing.Any] = False):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assign value
    :type value: typing.Optional[typing.Any]
    :param relative: Relative, Apply relative to the current value (delta)
    :type relative: typing.Union[bool, typing.Any]
    '''

    pass


def context_set_string(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       data_path: typing.Union[str, typing.Any] = "",
                       value: typing.Union[str, typing.Any] = ""):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assign value
    :type value: typing.Union[str, typing.Any]
    '''

    pass


def context_set_value(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      value: typing.Union[str, typing.Any] = ""):
    ''' Set a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value: Value, Assignment value (as a string)
    :type value: typing.Union[str, typing.Any]
    '''

    pass


def context_toggle(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   data_path: typing.Union[str, typing.Any] = "",
                   module: typing.Union[str, typing.Any] = ""):
    ''' Toggle a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param module: Module, Optionally override the context with a module
    :type module: typing.Union[str, typing.Any]
    '''

    pass


def context_toggle_enum(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        data_path: typing.Union[str, typing.Any] = "",
                        value_1: typing.Union[str, typing.Any] = "",
                        value_2: typing.Union[str, typing.Any] = ""):
    ''' Toggle a context value

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Context Attributes, RNA context string
    :type data_path: typing.Union[str, typing.Any]
    :param value_1: Value, Toggle enum
    :type value_1: typing.Union[str, typing.Any]
    :param value_2: Value, Toggle enum
    :type value_2: typing.Union[str, typing.Any]
    '''

    pass


def debug_menu(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               debug_value: typing.Optional[typing.Any] = 0):
    ''' Open a popup to set the debug level

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param debug_value: Debug Value
    :type debug_value: typing.Optional[typing.Any]
    '''

    pass


def doc_view(override_context: typing.Union[typing.
                                            Dict, 'bpy.types.Context'] = None,
             execution_context: typing.Union[str, int] = None,
             undo: typing.Optional[bool] = None,
             *,
             doc_id: typing.Union[str, typing.Any] = ""):
    ''' Open online reference docs in a web browser

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param doc_id: Doc ID
    :type doc_id: typing.Union[str, typing.Any]
    '''

    pass


def doc_view_manual(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    doc_id: typing.Union[str, typing.Any] = ""):
    ''' Load online manual

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param doc_id: Doc ID
    :type doc_id: typing.Union[str, typing.Any]
    '''

    pass


def doc_view_manual_ui_context(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None):
    ''' View a context based online manual in a web browser

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def drop_blend_file(override_context: typing.
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


def gpencil_export_pdf(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        filepath: typing.Union[str, typing.Any] = "",
        check_existing: typing.Union[bool, typing.Any] = True,
        filter_blender: typing.Union[bool, typing.Any] = False,
        filter_backup: typing.Union[bool, typing.Any] = False,
        filter_image: typing.Union[bool, typing.Any] = False,
        filter_movie: typing.Union[bool, typing.Any] = False,
        filter_python: typing.Union[bool, typing.Any] = False,
        filter_font: typing.Union[bool, typing.Any] = False,
        filter_sound: typing.Union[bool, typing.Any] = False,
        filter_text: typing.Union[bool, typing.Any] = False,
        filter_archive: typing.Union[bool, typing.Any] = False,
        filter_btx: typing.Union[bool, typing.Any] = False,
        filter_collada: typing.Union[bool, typing.Any] = False,
        filter_alembic: typing.Union[bool, typing.Any] = False,
        filter_usd: typing.Union[bool, typing.Any] = False,
        filter_volume: typing.Union[bool, typing.Any] = False,
        filter_folder: typing.Union[bool, typing.Any] = False,
        filter_blenlib: typing.Union[bool, typing.Any] = False,
        filemode: typing.Optional[typing.Any] = 8,
        display_type: typing.Optional[typing.Any] = 'DEFAULT',
        sort_method: typing.Union[str, int, typing.Any] = '',
        use_fill: typing.Union[bool, typing.Any] = True,
        selected_object_type: typing.Optional[typing.Any] = 'SELECTED',
        stroke_sample: typing.Optional[typing.Any] = 0.0,
        use_normalized_thickness: typing.Union[bool, typing.Any] = False,
        frame_mode: typing.Optional[typing.Any] = 'ACTIVE'):
    ''' Export grease pencil to PDF

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param use_fill: Fill, Export strokes with fill enabled
    :type use_fill: typing.Union[bool, typing.Any]
    :param selected_object_type: Object, Which objects to include in the export * ``ACTIVE`` Active, Include only the active object. * ``SELECTED`` Selected, Include selected objects. * ``VISIBLE`` Visible, Include all visible objects.
    :type selected_object_type: typing.Optional[typing.Any]
    :param stroke_sample: Sampling, Precision of stroke sampling. Low values mean a more precise result, and zero disables sampling
    :type stroke_sample: typing.Optional[typing.Any]
    :param use_normalized_thickness: Normalize, Export strokes with constant thickness
    :type use_normalized_thickness: typing.Union[bool, typing.Any]
    :param frame_mode: Frames, Which frames to include in the export * ``ACTIVE`` Active, Include only active frame. * ``SELECTED`` Selected, Include selected frames.
    :type frame_mode: typing.Optional[typing.Any]
    '''

    pass


def gpencil_export_svg(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        filepath: typing.Union[str, typing.Any] = "",
        check_existing: typing.Union[bool, typing.Any] = True,
        filter_blender: typing.Union[bool, typing.Any] = False,
        filter_backup: typing.Union[bool, typing.Any] = False,
        filter_image: typing.Union[bool, typing.Any] = False,
        filter_movie: typing.Union[bool, typing.Any] = False,
        filter_python: typing.Union[bool, typing.Any] = False,
        filter_font: typing.Union[bool, typing.Any] = False,
        filter_sound: typing.Union[bool, typing.Any] = False,
        filter_text: typing.Union[bool, typing.Any] = False,
        filter_archive: typing.Union[bool, typing.Any] = False,
        filter_btx: typing.Union[bool, typing.Any] = False,
        filter_collada: typing.Union[bool, typing.Any] = False,
        filter_alembic: typing.Union[bool, typing.Any] = False,
        filter_usd: typing.Union[bool, typing.Any] = False,
        filter_volume: typing.Union[bool, typing.Any] = False,
        filter_folder: typing.Union[bool, typing.Any] = False,
        filter_blenlib: typing.Union[bool, typing.Any] = False,
        filemode: typing.Optional[typing.Any] = 8,
        display_type: typing.Optional[typing.Any] = 'DEFAULT',
        sort_method: typing.Union[str, int, typing.Any] = '',
        use_fill: typing.Union[bool, typing.Any] = True,
        selected_object_type: typing.Optional[typing.Any] = 'SELECTED',
        stroke_sample: typing.Optional[typing.Any] = 0.0,
        use_normalized_thickness: typing.Union[bool, typing.Any] = False,
        use_clip_camera: typing.Union[bool, typing.Any] = False):
    ''' Export grease pencil to SVG

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param use_fill: Fill, Export strokes with fill enabled
    :type use_fill: typing.Union[bool, typing.Any]
    :param selected_object_type: Object, Which objects to include in the export * ``ACTIVE`` Active, Include only the active object. * ``SELECTED`` Selected, Include selected objects. * ``VISIBLE`` Visible, Include all visible objects.
    :type selected_object_type: typing.Optional[typing.Any]
    :param stroke_sample: Sampling, Precision of stroke sampling. Low values mean a more precise result, and zero disables sampling
    :type stroke_sample: typing.Optional[typing.Any]
    :param use_normalized_thickness: Normalize, Export strokes with constant thickness
    :type use_normalized_thickness: typing.Union[bool, typing.Any]
    :param use_clip_camera: Clip Camera, Clip drawings to camera size when export in camera view
    :type use_clip_camera: typing.Union[bool, typing.Any]
    '''

    pass


def gpencil_import_svg(override_context: typing.
                       Union[typing.Dict, 'bpy.types.Context'] = None,
                       execution_context: typing.Union[str, int] = None,
                       undo: typing.Optional[bool] = None,
                       *,
                       filepath: typing.Union[str, typing.Any] = "",
                       filter_blender: typing.Union[bool, typing.Any] = False,
                       filter_backup: typing.Union[bool, typing.Any] = False,
                       filter_image: typing.Union[bool, typing.Any] = False,
                       filter_movie: typing.Union[bool, typing.Any] = False,
                       filter_python: typing.Union[bool, typing.Any] = False,
                       filter_font: typing.Union[bool, typing.Any] = False,
                       filter_sound: typing.Union[bool, typing.Any] = False,
                       filter_text: typing.Union[bool, typing.Any] = False,
                       filter_archive: typing.Union[bool, typing.Any] = False,
                       filter_btx: typing.Union[bool, typing.Any] = False,
                       filter_collada: typing.Union[bool, typing.Any] = False,
                       filter_alembic: typing.Union[bool, typing.Any] = False,
                       filter_usd: typing.Union[bool, typing.Any] = False,
                       filter_volume: typing.Union[bool, typing.Any] = False,
                       filter_folder: typing.Union[bool, typing.Any] = False,
                       filter_blenlib: typing.Union[bool, typing.Any] = False,
                       filemode: typing.Optional[typing.Any] = 8,
                       relative_path: typing.Union[bool, typing.Any] = True,
                       display_type: typing.Optional[typing.Any] = 'DEFAULT',
                       sort_method: typing.Union[str, int, typing.Any] = '',
                       resolution: typing.Optional[typing.Any] = 10,
                       scale: typing.Optional[typing.Any] = 10.0):
    ''' Import SVG into grease pencil

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
    :param relative_path: Relative Path, Select the file relative to the blend file
    :type relative_path: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param resolution: Resolution, Resolution of the generated strokes
    :type resolution: typing.Optional[typing.Any]
    :param scale: Scale, Scale of the final strokes
    :type scale: typing.Optional[typing.Any]
    '''

    pass


def interface_theme_preset_add(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        name: typing.Union[str, typing.Any] = "",
        remove_name: typing.Union[bool, typing.Any] = False,
        remove_active: typing.Union[bool, typing.Any] = False):
    ''' Add or remove a theme preset

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the preset, used to make the path name
    :type name: typing.Union[str, typing.Any]
    :param remove_name: remove_name
    :type remove_name: typing.Union[bool, typing.Any]
    :param remove_active: remove_active
    :type remove_active: typing.Union[bool, typing.Any]
    '''

    pass


def keyconfig_preset_add(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        name: typing.Union[str, typing.Any] = "",
        remove_name: typing.Union[bool, typing.Any] = False,
        remove_active: typing.Union[bool, typing.Any] = False):
    ''' Add or remove a Key-config Preset

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the preset, used to make the path name
    :type name: typing.Union[str, typing.Any]
    :param remove_name: remove_name
    :type remove_name: typing.Union[bool, typing.Any]
    :param remove_active: remove_active
    :type remove_active: typing.Union[bool, typing.Any]
    '''

    pass


def lib_reload(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               library: typing.Union[str, typing.Any] = "",
               filepath: typing.Union[str, typing.Any] = "",
               directory: typing.Union[str, typing.Any] = "",
               filename: typing.Union[str, typing.Any] = "",
               hide_props_region: typing.Union[bool, typing.Any] = True,
               filter_blender: typing.Union[bool, typing.Any] = True,
               filter_backup: typing.Union[bool, typing.Any] = False,
               filter_image: typing.Union[bool, typing.Any] = False,
               filter_movie: typing.Union[bool, typing.Any] = False,
               filter_python: typing.Union[bool, typing.Any] = False,
               filter_font: typing.Union[bool, typing.Any] = False,
               filter_sound: typing.Union[bool, typing.Any] = False,
               filter_text: typing.Union[bool, typing.Any] = False,
               filter_archive: typing.Union[bool, typing.Any] = False,
               filter_btx: typing.Union[bool, typing.Any] = False,
               filter_collada: typing.Union[bool, typing.Any] = False,
               filter_alembic: typing.Union[bool, typing.Any] = False,
               filter_usd: typing.Union[bool, typing.Any] = False,
               filter_volume: typing.Union[bool, typing.Any] = False,
               filter_folder: typing.Union[bool, typing.Any] = True,
               filter_blenlib: typing.Union[bool, typing.Any] = False,
               filemode: typing.Optional[typing.Any] = 8,
               relative_path: typing.Union[bool, typing.Any] = True,
               display_type: typing.Optional[typing.Any] = 'DEFAULT',
               sort_method: typing.Union[str, int, typing.Any] = ''):
    ''' Reload the given library

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param library: Library, Library to reload
    :type library: typing.Union[str, typing.Any]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param directory: Directory, Directory of the file
    :type directory: typing.Union[str, typing.Any]
    :param filename: File Name, Name of the file
    :type filename: typing.Union[str, typing.Any]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
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
    :param relative_path: Relative Path, Select the file relative to the blend file
    :type relative_path: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    '''

    pass


def lib_relocate(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 library: typing.Union[str, typing.Any] = "",
                 filepath: typing.Union[str, typing.Any] = "",
                 directory: typing.Union[str, typing.Any] = "",
                 filename: typing.Union[str, typing.Any] = "",
                 files: typing.Optional[bpy.types.bpy_prop_collection[
                     'bpy.types.OperatorFileListElement']] = None,
                 hide_props_region: typing.Union[bool, typing.Any] = True,
                 filter_blender: typing.Union[bool, typing.Any] = True,
                 filter_backup: typing.Union[bool, typing.Any] = False,
                 filter_image: typing.Union[bool, typing.Any] = False,
                 filter_movie: typing.Union[bool, typing.Any] = False,
                 filter_python: typing.Union[bool, typing.Any] = False,
                 filter_font: typing.Union[bool, typing.Any] = False,
                 filter_sound: typing.Union[bool, typing.Any] = False,
                 filter_text: typing.Union[bool, typing.Any] = False,
                 filter_archive: typing.Union[bool, typing.Any] = False,
                 filter_btx: typing.Union[bool, typing.Any] = False,
                 filter_collada: typing.Union[bool, typing.Any] = False,
                 filter_alembic: typing.Union[bool, typing.Any] = False,
                 filter_usd: typing.Union[bool, typing.Any] = False,
                 filter_volume: typing.Union[bool, typing.Any] = False,
                 filter_folder: typing.Union[bool, typing.Any] = True,
                 filter_blenlib: typing.Union[bool, typing.Any] = False,
                 filemode: typing.Optional[typing.Any] = 8,
                 relative_path: typing.Union[bool, typing.Any] = True,
                 display_type: typing.Optional[typing.Any] = 'DEFAULT',
                 sort_method: typing.Union[str, int, typing.Any] = ''):
    ''' Relocate the given library to one or several others

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param library: Library, Library to relocate
    :type library: typing.Union[str, typing.Any]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param directory: Directory, Directory of the file
    :type directory: typing.Union[str, typing.Any]
    :param filename: File Name, Name of the file
    :type filename: typing.Union[str, typing.Any]
    :param files: Files
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
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
    :param relative_path: Relative Path, Select the file relative to the blend file
    :type relative_path: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    '''

    pass


def link(override_context: typing.Union[typing.
                                        Dict, 'bpy.types.Context'] = None,
         execution_context: typing.Union[str, int] = None,
         undo: typing.Optional[bool] = None,
         *,
         filepath: typing.Union[str, typing.Any] = "",
         directory: typing.Union[str, typing.Any] = "",
         filename: typing.Union[str, typing.Any] = "",
         files: typing.Optional[bpy.types.bpy_prop_collection[
             'bpy.types.OperatorFileListElement']] = None,
         filter_blender: typing.Union[bool, typing.Any] = True,
         filter_backup: typing.Union[bool, typing.Any] = False,
         filter_image: typing.Union[bool, typing.Any] = False,
         filter_movie: typing.Union[bool, typing.Any] = False,
         filter_python: typing.Union[bool, typing.Any] = False,
         filter_font: typing.Union[bool, typing.Any] = False,
         filter_sound: typing.Union[bool, typing.Any] = False,
         filter_text: typing.Union[bool, typing.Any] = False,
         filter_archive: typing.Union[bool, typing.Any] = False,
         filter_btx: typing.Union[bool, typing.Any] = False,
         filter_collada: typing.Union[bool, typing.Any] = False,
         filter_alembic: typing.Union[bool, typing.Any] = False,
         filter_usd: typing.Union[bool, typing.Any] = False,
         filter_volume: typing.Union[bool, typing.Any] = False,
         filter_folder: typing.Union[bool, typing.Any] = True,
         filter_blenlib: typing.Union[bool, typing.Any] = True,
         filemode: typing.Optional[typing.Any] = 1,
         relative_path: typing.Union[bool, typing.Any] = True,
         display_type: typing.Optional[typing.Any] = 'DEFAULT',
         sort_method: typing.Union[str, int, typing.Any] = '',
         link: typing.Union[bool, typing.Any] = True,
         autoselect: typing.Union[bool, typing.Any] = True,
         active_collection: typing.Union[bool, typing.Any] = True,
         instance_collections: typing.Union[bool, typing.Any] = True,
         instance_object_data: typing.Union[bool, typing.Any] = True):
    ''' Link from a Library .blend file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param directory: Directory, Directory of the file
    :type directory: typing.Union[str, typing.Any]
    :param filename: File Name, Name of the file
    :type filename: typing.Union[str, typing.Any]
    :param files: Files
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
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
    :param relative_path: Relative Path, Select the file relative to the blend file
    :type relative_path: typing.Union[bool, typing.Any]
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param link: Link, Link the objects or data-blocks rather than appending
    :type link: typing.Union[bool, typing.Any]
    :param autoselect: Select, Select new objects
    :type autoselect: typing.Union[bool, typing.Any]
    :param active_collection: Active Collection, Put new objects on the active collection
    :type active_collection: typing.Union[bool, typing.Any]
    :param instance_collections: Instance Collections, Create instances for collections, rather than adding them directly to the scene
    :type instance_collections: typing.Union[bool, typing.Any]
    :param instance_object_data: Instance Object Data, Create instances for object data which are not referenced by any objects
    :type instance_object_data: typing.Union[bool, typing.Any]
    '''

    pass


def memory_statistics(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Print memory statistics to the console

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def open_mainfile(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  filepath: typing.Union[str, typing.Any] = "",
                  hide_props_region: typing.Union[bool, typing.Any] = True,
                  filter_blender: typing.Union[bool, typing.Any] = True,
                  filter_backup: typing.Union[bool, typing.Any] = False,
                  filter_image: typing.Union[bool, typing.Any] = False,
                  filter_movie: typing.Union[bool, typing.Any] = False,
                  filter_python: typing.Union[bool, typing.Any] = False,
                  filter_font: typing.Union[bool, typing.Any] = False,
                  filter_sound: typing.Union[bool, typing.Any] = False,
                  filter_text: typing.Union[bool, typing.Any] = False,
                  filter_archive: typing.Union[bool, typing.Any] = False,
                  filter_btx: typing.Union[bool, typing.Any] = False,
                  filter_collada: typing.Union[bool, typing.Any] = False,
                  filter_alembic: typing.Union[bool, typing.Any] = False,
                  filter_usd: typing.Union[bool, typing.Any] = False,
                  filter_volume: typing.Union[bool, typing.Any] = False,
                  filter_folder: typing.Union[bool, typing.Any] = True,
                  filter_blenlib: typing.Union[bool, typing.Any] = False,
                  filemode: typing.Optional[typing.Any] = 8,
                  display_type: typing.Optional[typing.Any] = 'DEFAULT',
                  sort_method: typing.Union[str, int, typing.Any] = '',
                  load_ui: typing.Union[bool, typing.Any] = True,
                  use_scripts: typing.Union[bool, typing.Any] = True,
                  display_file_selector: typing.Union[bool, typing.Any] = True,
                  state: typing.Optional[typing.Any] = 0):
    ''' Open a Blender file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param load_ui: Load UI, Load user interface setup in the .blend file
    :type load_ui: typing.Union[bool, typing.Any]
    :param use_scripts: Trusted Source, Allow .blend file to execute scripts automatically, default available from system preferences
    :type use_scripts: typing.Union[bool, typing.Any]
    :param display_file_selector: Display File Selector
    :type display_file_selector: typing.Union[bool, typing.Any]
    :param state: State
    :type state: typing.Optional[typing.Any]
    '''

    pass


def operator_cheat_sheet(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None):
    ''' List all the operators in a text-block, useful for scripting :file: `startup/bl_operators/wm.py\:1742 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/wm.py$1742>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def operator_defaults(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Set the active operator to its default values

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def operator_pie_enum(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      prop_string: typing.Union[str, typing.Any] = ""):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__.

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Operator, Operator name (in python as string)
    :type data_path: typing.Union[str, typing.Any]
    :param prop_string: Property, Property name (as a string)
    :type prop_string: typing.Union[str, typing.Any]
    '''

    pass


def operator_preset_add(override_context: typing.
                        Union[typing.Dict, 'bpy.types.Context'] = None,
                        execution_context: typing.Union[str, int] = None,
                        undo: typing.Optional[bool] = None,
                        *,
                        name: typing.Union[str, typing.Any] = "",
                        remove_name: typing.Union[bool, typing.Any] = False,
                        remove_active: typing.Union[bool, typing.Any] = False,
                        operator: typing.Union[str, typing.Any] = ""):
    ''' Add or remove an Operator Preset

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Name, Name of the preset, used to make the path name
    :type name: typing.Union[str, typing.Any]
    :param remove_name: remove_name
    :type remove_name: typing.Union[bool, typing.Any]
    :param remove_active: remove_active
    :type remove_active: typing.Union[bool, typing.Any]
    :param operator: Operator
    :type operator: typing.Union[str, typing.Any]
    '''

    pass


def owner_disable(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  owner_id: typing.Union[str, typing.Any] = ""):
    ''' Enable workspace owner ID

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param owner_id: UI Tag
    :type owner_id: typing.Union[str, typing.Any]
    '''

    pass


def owner_enable(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 owner_id: typing.Union[str, typing.Any] = ""):
    ''' Enable workspace owner ID

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param owner_id: UI Tag
    :type owner_id: typing.Union[str, typing.Any]
    '''

    pass


def path_open(override_context: typing.Union[typing.
                                             Dict, 'bpy.types.Context'] = None,
              execution_context: typing.Union[str, int] = None,
              undo: typing.Optional[bool] = None,
              *,
              filepath: typing.Union[str, typing.Any] = ""):
    ''' Open a path in a file browser

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    '''

    pass


def previews_batch_clear(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        files: typing.Optional[bpy.types.bpy_prop_collection[
            'bpy.types.OperatorFileListElement']] = None,
        directory: typing.Union[str, typing.Any] = "",
        filter_blender: typing.Union[bool, typing.Any] = True,
        filter_folder: typing.Union[bool, typing.Any] = True,
        use_scenes: typing.Union[bool, typing.Any] = True,
        use_collections: typing.Union[bool, typing.Any] = True,
        use_objects: typing.Union[bool, typing.Any] = True,
        use_intern_data: typing.Union[bool, typing.Any] = True,
        use_trusted: typing.Union[bool, typing.Any] = False,
        use_backups: typing.Union[bool, typing.Any] = True):
    ''' Clear selected .blend file's previews

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param files: files
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
    :param directory: directory
    :type directory: typing.Union[str, typing.Any]
    :param filter_blender: filter_blender
    :type filter_blender: typing.Union[bool, typing.Any]
    :param filter_folder: filter_folder
    :type filter_folder: typing.Union[bool, typing.Any]
    :param use_scenes: Scenes, Clear scenes' previews
    :type use_scenes: typing.Union[bool, typing.Any]
    :param use_collections: Collections, Clear collections' previews
    :type use_collections: typing.Union[bool, typing.Any]
    :param use_objects: Objects, Clear objects' previews
    :type use_objects: typing.Union[bool, typing.Any]
    :param use_intern_data: Materials & Textures, Clear 'internal' previews (materials, textures, images, etc.)
    :type use_intern_data: typing.Union[bool, typing.Any]
    :param use_trusted: Trusted Blend Files, Enable python evaluation for selected files
    :type use_trusted: typing.Union[bool, typing.Any]
    :param use_backups: Save Backups, Keep a backup (.blend1) version of the files when saving with cleared previews
    :type use_backups: typing.Union[bool, typing.Any]
    '''

    pass


def previews_batch_generate(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        files: typing.Optional[bpy.types.bpy_prop_collection[
            'bpy.types.OperatorFileListElement']] = None,
        directory: typing.Union[str, typing.Any] = "",
        filter_blender: typing.Union[bool, typing.Any] = True,
        filter_folder: typing.Union[bool, typing.Any] = True,
        use_scenes: typing.Union[bool, typing.Any] = True,
        use_collections: typing.Union[bool, typing.Any] = True,
        use_objects: typing.Union[bool, typing.Any] = True,
        use_intern_data: typing.Union[bool, typing.Any] = True,
        use_trusted: typing.Union[bool, typing.Any] = False,
        use_backups: typing.Union[bool, typing.Any] = True):
    ''' Generate selected .blend file's previews

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param files: files
    :type files: typing.Optional[bpy.types.bpy_prop_collection['bpy.types.OperatorFileListElement']]
    :param directory: directory
    :type directory: typing.Union[str, typing.Any]
    :param filter_blender: filter_blender
    :type filter_blender: typing.Union[bool, typing.Any]
    :param filter_folder: filter_folder
    :type filter_folder: typing.Union[bool, typing.Any]
    :param use_scenes: Scenes, Generate scenes' previews
    :type use_scenes: typing.Union[bool, typing.Any]
    :param use_collections: Collections, Generate collections' previews
    :type use_collections: typing.Union[bool, typing.Any]
    :param use_objects: Objects, Generate objects' previews
    :type use_objects: typing.Union[bool, typing.Any]
    :param use_intern_data: Materials & Textures, Generate 'internal' previews (materials, textures, images, etc.)
    :type use_intern_data: typing.Union[bool, typing.Any]
    :param use_trusted: Trusted Blend Files, Enable python evaluation for selected files
    :type use_trusted: typing.Union[bool, typing.Any]
    :param use_backups: Save Backups, Keep a backup (.blend1) version of the files when saving with generated previews
    :type use_backups: typing.Union[bool, typing.Any]
    '''

    pass


def previews_clear(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   id_type: typing.Optional[typing.Any] = {}):
    ''' Clear data-block previews (only for some types like objects, materials, textures, etc.)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param id_type: Data-Block Type, Which data-block previews to clear * ``ALL`` All Types. * ``GEOMETRY`` All Geometry Types, Clear previews for scenes, collections and objects. * ``SHADING`` All Shading Types, Clear previews for materials, lights, worlds, textures and images. * ``SCENE`` Scenes. * ``COLLECTION`` Collections. * ``OBJECT`` Objects. * ``MATERIAL`` Materials. * ``LIGHT`` Lights. * ``WORLD`` Worlds. * ``TEXTURE`` Textures. * ``IMAGE`` Images.
    :type id_type: typing.Optional[typing.Any]
    '''

    pass


def previews_ensure(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None):
    ''' Ensure data-block previews are available and up-to-date (to be saved in .blend file, only for some types like materials, textures, etc.)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def properties_add(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   data_path: typing.Union[str, typing.Any] = ""):
    ''' Add your own property to the data-block

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Property Edit, Property data_path edit
    :type data_path: typing.Union[str, typing.Any]
    '''

    pass


def properties_context_change(override_context: typing.
                              Union[typing.Dict, 'bpy.types.Context'] = None,
                              execution_context: typing.Union[str, int] = None,
                              undo: typing.Optional[bool] = None,
                              *,
                              context: typing.Union[str, typing.Any] = ""):
    ''' Jump to a different tab inside the properties editor

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param context: Context
    :type context: typing.Union[str, typing.Any]
    '''

    pass


def properties_edit(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        data_path: typing.Union[str, typing.Any] = "",
        property: typing.Union[str, typing.Any] = "",
        value: typing.Union[str, typing.Any] = "",
        default: typing.Union[str, typing.Any] = "",
        min: typing.Optional[typing.Any] = -10000,
        max: typing.Optional[typing.Any] = 10000.0,
        use_soft_limits: typing.Union[bool, typing.Any] = False,
        is_overridable_library: typing.Union[bool, typing.Any] = False,
        soft_min: typing.Optional[typing.Any] = -10000,
        soft_max: typing.Optional[typing.Any] = 10000.0,
        description: typing.Union[str, typing.Any] = "",
        subtype: typing.Union[str, int, typing.Any] = ''):
    ''' Edit the attributes of the property

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Property Edit, Property data_path edit
    :type data_path: typing.Union[str, typing.Any]
    :param property: Property Name, Property name edit
    :type property: typing.Union[str, typing.Any]
    :param value: Property Value, Property value edit
    :type value: typing.Union[str, typing.Any]
    :param default: Default Value, Default value of the property. Important for NLA mixing
    :type default: typing.Union[str, typing.Any]
    :param min: Min, Minimum value of the property
    :type min: typing.Optional[typing.Any]
    :param max: Max, Maximum value of the property
    :type max: typing.Optional[typing.Any]
    :param use_soft_limits: Use Soft Limits, Limits the Property Value slider to a range, values outside the range must be inputted numerically
    :type use_soft_limits: typing.Union[bool, typing.Any]
    :param is_overridable_library: Is Library Overridable, Allow the property to be overridden when the data-block is linked
    :type is_overridable_library: typing.Union[bool, typing.Any]
    :param soft_min: Min, Minimum value of the property
    :type soft_min: typing.Optional[typing.Any]
    :param soft_max: Max, Maximum value of the property
    :type soft_max: typing.Optional[typing.Any]
    :param description: Tooltip
    :type description: typing.Union[str, typing.Any]
    :param subtype: Subtype
    :type subtype: typing.Union[str, int, typing.Any]
    '''

    pass


def properties_remove(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      data_path: typing.Union[str, typing.Any] = "",
                      property: typing.Union[str, typing.Any] = ""):
    ''' Internal use (edit a property data_path)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path: Property Edit, Property data_path edit
    :type data_path: typing.Union[str, typing.Any]
    :param property: Property Name, Property name edit
    :type property: typing.Union[str, typing.Any]
    '''

    pass


def quit_blender(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None):
    ''' Quit Blender

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def radial_control(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        data_path_primary: typing.Union[str, typing.Any] = "",
        data_path_secondary: typing.Union[str, typing.Any] = "",
        use_secondary: typing.Union[str, typing.Any] = "",
        rotation_path: typing.Union[str, typing.Any] = "",
        color_path: typing.Union[str, typing.Any] = "",
        fill_color_path: typing.Union[str, typing.Any] = "",
        fill_color_override_path: typing.Union[str, typing.Any] = "",
        fill_color_override_test_path: typing.Union[str, typing.Any] = "",
        zoom_path: typing.Union[str, typing.Any] = "",
        image_id: typing.Union[str, typing.Any] = "",
        secondary_tex: typing.Union[bool, typing.Any] = False,
        release_confirm: typing.Union[bool, typing.Any] = False):
    ''' Set some size property (e.g. brush size) with mouse wheel

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param data_path_primary: Primary Data Path, Primary path of property to be set by the radial control
    :type data_path_primary: typing.Union[str, typing.Any]
    :param data_path_secondary: Secondary Data Path, Secondary path of property to be set by the radial control
    :type data_path_secondary: typing.Union[str, typing.Any]
    :param use_secondary: Use Secondary, Path of property to select between the primary and secondary data paths
    :type use_secondary: typing.Union[str, typing.Any]
    :param rotation_path: Rotation Path, Path of property used to rotate the texture display
    :type rotation_path: typing.Union[str, typing.Any]
    :param color_path: Color Path, Path of property used to set the color of the control
    :type color_path: typing.Union[str, typing.Any]
    :param fill_color_path: Fill Color Path, Path of property used to set the fill color of the control
    :type fill_color_path: typing.Union[str, typing.Any]
    :param fill_color_override_path: Fill Color Override Path
    :type fill_color_override_path: typing.Union[str, typing.Any]
    :param fill_color_override_test_path: Fill Color Override Test
    :type fill_color_override_test_path: typing.Union[str, typing.Any]
    :param zoom_path: Zoom Path, Path of property used to set the zoom level for the control
    :type zoom_path: typing.Union[str, typing.Any]
    :param image_id: Image ID, Path of ID that is used to generate an image for the control
    :type image_id: typing.Union[str, typing.Any]
    :param secondary_tex: Secondary Texture, Tweak brush secondary/mask texture
    :type secondary_tex: typing.Union[bool, typing.Any]
    :param release_confirm: Confirm On Release, Finish operation on key release
    :type release_confirm: typing.Union[bool, typing.Any]
    '''

    pass


def read_factory_settings(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        app_template: typing.Union[str, typing.Any] = "Template",
        use_empty: typing.Union[bool, typing.Any] = False):
    ''' Load factory default startup file and preferences. To make changes permanent, use "Save Startup File" and "Save Preferences"

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :type app_template: typing.Union[str, typing.Any]
    :param use_empty: Empty
    :type use_empty: typing.Union[bool, typing.Any]
    '''

    pass


def read_factory_userpref(override_context: typing.
                          Union[typing.Dict, 'bpy.types.Context'] = None,
                          execution_context: typing.Union[str, int] = None,
                          undo: typing.Optional[bool] = None):
    ''' Load factory default preferences. To make changes to preferences permanent, use "Save Preferences"

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def read_history(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None):
    ''' Reloads history and bookmarks

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def read_homefile(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  filepath: typing.Union[str, typing.Any] = "",
                  load_ui: typing.Union[bool, typing.Any] = True,
                  use_splash: typing.Union[bool, typing.Any] = False,
                  use_factory_startup: typing.Union[bool, typing.Any] = False,
                  app_template: typing.Union[str, typing.Any] = "Template",
                  use_empty: typing.Union[bool, typing.Any] = False):
    ''' Open the default file (doesn't save the current file)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to an alternative start-up file
    :type filepath: typing.Union[str, typing.Any]
    :param load_ui: Load UI, Load user interface setup from the .blend file
    :type load_ui: typing.Union[bool, typing.Any]
    :param use_splash: Splash
    :type use_splash: typing.Union[bool, typing.Any]
    :param use_factory_startup: Factory Startup
    :type use_factory_startup: typing.Union[bool, typing.Any]
    :type app_template: typing.Union[str, typing.Any]
    :param use_empty: Empty
    :type use_empty: typing.Union[bool, typing.Any]
    '''

    pass


def read_userpref(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None):
    ''' Load last saved preferences

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def recover_auto_save(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        filepath: typing.Union[str, typing.Any] = "",
        hide_props_region: typing.Union[bool, typing.Any] = True,
        filter_blender: typing.Union[bool, typing.Any] = True,
        filter_backup: typing.Union[bool, typing.Any] = False,
        filter_image: typing.Union[bool, typing.Any] = False,
        filter_movie: typing.Union[bool, typing.Any] = False,
        filter_python: typing.Union[bool, typing.Any] = False,
        filter_font: typing.Union[bool, typing.Any] = False,
        filter_sound: typing.Union[bool, typing.Any] = False,
        filter_text: typing.Union[bool, typing.Any] = False,
        filter_archive: typing.Union[bool, typing.Any] = False,
        filter_btx: typing.Union[bool, typing.Any] = False,
        filter_collada: typing.Union[bool, typing.Any] = False,
        filter_alembic: typing.Union[bool, typing.Any] = False,
        filter_usd: typing.Union[bool, typing.Any] = False,
        filter_volume: typing.Union[bool, typing.Any] = False,
        filter_folder: typing.Union[bool, typing.Any] = False,
        filter_blenlib: typing.Union[bool, typing.Any] = False,
        filemode: typing.Optional[typing.Any] = 8,
        display_type: typing.Optional[typing.Any] = 'LIST_VERTICAL',
        sort_method: typing.Union[str, int, typing.Any] = '',
        use_scripts: typing.Union[bool, typing.Any] = True):
    ''' Open an automatically saved file to recover it

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param use_scripts: Trusted Source, Allow .blend file to execute scripts automatically, default available from system preferences
    :type use_scripts: typing.Union[bool, typing.Any]
    '''

    pass


def recover_last_session(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None,
                         *,
                         use_scripts: typing.Union[bool, typing.Any] = True):
    ''' Open the last closed file ("quit.blend")

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param use_scripts: Trusted Source, Allow .blend file to execute scripts automatically, default available from system preferences
    :type use_scripts: typing.Union[bool, typing.Any]
    '''

    pass


def redraw_timer(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None,
                 *,
                 type: typing.Optional[typing.Any] = 'DRAW',
                 iterations: typing.Optional[typing.Any] = 10,
                 time_limit: typing.Optional[typing.Any] = 0.0):
    ''' Simple redraw timer to test the speed of updating the interface

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Type * ``DRAW`` Draw Region, Draw region. * ``DRAW_SWAP`` Draw Region & Swap, Draw region and swap. * ``DRAW_WIN`` Draw Window, Draw window. * ``DRAW_WIN_SWAP`` Draw Window & Swap, Draw window and swap. * ``ANIM_STEP`` Animation Step, Animation steps. * ``ANIM_PLAY`` Animation Play, Animation playback. * ``UNDO`` Undo/Redo, Undo and redo.
    :type type: typing.Optional[typing.Any]
    :param iterations: Iterations, Number of times to redraw
    :type iterations: typing.Optional[typing.Any]
    :param time_limit: Time Limit, Seconds to run the test for (override iterations)
    :type time_limit: typing.Optional[typing.Any]
    '''

    pass


def revert_mainfile(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    use_scripts: typing.Union[bool, typing.Any] = True):
    ''' Reload the saved file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param use_scripts: Trusted Source, Allow .blend file to execute scripts automatically, default available from system preferences
    :type use_scripts: typing.Union[bool, typing.Any]
    '''

    pass


def save_as_mainfile(override_context: typing.
                     Union[typing.Dict, 'bpy.types.Context'] = None,
                     execution_context: typing.Union[str, int] = None,
                     undo: typing.Optional[bool] = None,
                     *,
                     filepath: typing.Union[str, typing.Any] = "",
                     hide_props_region: typing.Union[bool, typing.Any] = True,
                     check_existing: typing.Union[bool, typing.Any] = True,
                     filter_blender: typing.Union[bool, typing.Any] = True,
                     filter_backup: typing.Union[bool, typing.Any] = False,
                     filter_image: typing.Union[bool, typing.Any] = False,
                     filter_movie: typing.Union[bool, typing.Any] = False,
                     filter_python: typing.Union[bool, typing.Any] = False,
                     filter_font: typing.Union[bool, typing.Any] = False,
                     filter_sound: typing.Union[bool, typing.Any] = False,
                     filter_text: typing.Union[bool, typing.Any] = False,
                     filter_archive: typing.Union[bool, typing.Any] = False,
                     filter_btx: typing.Union[bool, typing.Any] = False,
                     filter_collada: typing.Union[bool, typing.Any] = False,
                     filter_alembic: typing.Union[bool, typing.Any] = False,
                     filter_usd: typing.Union[bool, typing.Any] = False,
                     filter_volume: typing.Union[bool, typing.Any] = False,
                     filter_folder: typing.Union[bool, typing.Any] = True,
                     filter_blenlib: typing.Union[bool, typing.Any] = False,
                     filemode: typing.Optional[typing.Any] = 8,
                     display_type: typing.Optional[typing.Any] = 'DEFAULT',
                     sort_method: typing.Union[str, int, typing.Any] = '',
                     compress: typing.Union[bool, typing.Any] = False,
                     relative_remap: typing.Union[bool, typing.Any] = True,
                     copy: typing.Union[bool, typing.Any] = False):
    ''' Save the current file in the desired location

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param compress: Compress, Write compressed .blend file
    :type compress: typing.Union[bool, typing.Any]
    :param relative_remap: Remap Relative, Remap relative paths when saving to a different directory
    :type relative_remap: typing.Union[bool, typing.Any]
    :param copy: Save Copy, Save a copy of the actual working state but does not make saved file active
    :type copy: typing.Union[bool, typing.Any]
    '''

    pass


def save_homefile(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None):
    ''' Make the current file the default .blend file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def save_mainfile(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None,
                  *,
                  filepath: typing.Union[str, typing.Any] = "",
                  hide_props_region: typing.Union[bool, typing.Any] = True,
                  check_existing: typing.Union[bool, typing.Any] = True,
                  filter_blender: typing.Union[bool, typing.Any] = True,
                  filter_backup: typing.Union[bool, typing.Any] = False,
                  filter_image: typing.Union[bool, typing.Any] = False,
                  filter_movie: typing.Union[bool, typing.Any] = False,
                  filter_python: typing.Union[bool, typing.Any] = False,
                  filter_font: typing.Union[bool, typing.Any] = False,
                  filter_sound: typing.Union[bool, typing.Any] = False,
                  filter_text: typing.Union[bool, typing.Any] = False,
                  filter_archive: typing.Union[bool, typing.Any] = False,
                  filter_btx: typing.Union[bool, typing.Any] = False,
                  filter_collada: typing.Union[bool, typing.Any] = False,
                  filter_alembic: typing.Union[bool, typing.Any] = False,
                  filter_usd: typing.Union[bool, typing.Any] = False,
                  filter_volume: typing.Union[bool, typing.Any] = False,
                  filter_folder: typing.Union[bool, typing.Any] = True,
                  filter_blenlib: typing.Union[bool, typing.Any] = False,
                  filemode: typing.Optional[typing.Any] = 8,
                  display_type: typing.Optional[typing.Any] = 'DEFAULT',
                  sort_method: typing.Union[str, int, typing.Any] = '',
                  compress: typing.Union[bool, typing.Any] = False,
                  relative_remap: typing.Union[bool, typing.Any] = False,
                  exit: typing.Union[bool, typing.Any] = False):
    ''' Save the current Blender file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param hide_props_region: Hide Operator Properties, Collapse the region displaying the operator settings
    :type hide_props_region: typing.Union[bool, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param compress: Compress, Write compressed .blend file
    :type compress: typing.Union[bool, typing.Any]
    :param relative_remap: Remap Relative, Remap relative paths when saving to a different directory
    :type relative_remap: typing.Union[bool, typing.Any]
    :param exit: Exit, Exit Blender after saving
    :type exit: typing.Union[bool, typing.Any]
    '''

    pass


def save_userpref(override_context: typing.
                  Union[typing.Dict, 'bpy.types.Context'] = None,
                  execution_context: typing.Union[str, int] = None,
                  undo: typing.Optional[bool] = None):
    ''' Make the current preferences default

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def search_menu(override_context: typing.
                Union[typing.Dict, 'bpy.types.Context'] = None,
                execution_context: typing.Union[str, int] = None,
                undo: typing.Optional[bool] = None):
    ''' Pop-up a search over all menus in the current context

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def search_operator(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None):
    ''' Pop-up a search over all available operators in current context

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def set_stereo_3d(
        override_context: typing.Union[typing.
                                       Dict, 'bpy.types.Context'] = None,
        execution_context: typing.Union[str, int] = None,
        undo: typing.Optional[bool] = None,
        *,
        display_mode: typing.Optional[typing.Any] = 'ANAGLYPH',
        anaglyph_type: typing.Optional[typing.Any] = 'RED_CYAN',
        interlace_type: typing.Optional[typing.Any] = 'ROW_INTERLEAVED',
        use_interlace_swap: typing.Union[bool, typing.Any] = False,
        use_sidebyside_crosseyed: typing.Union[bool, typing.Any] = False):
    ''' Toggle 3D stereo support for current window (or change the display mode)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param display_mode: Display Mode * ``ANAGLYPH`` Anaglyph, Render views for left and right eyes as two differently filtered colors in a single image (anaglyph glasses are required). * ``INTERLACE`` Interlace, Render views for left and right eyes interlaced in a single image (3D-ready monitor is required). * ``TIMESEQUENTIAL`` Time Sequential, Render alternate eyes (also known as page flip, quad buffer support in the graphic card is required). * ``SIDEBYSIDE`` Side-by-Side, Render views for left and right eyes side-by-side. * ``TOPBOTTOM`` Top-Bottom, Render views for left and right eyes one above another.
    :type display_mode: typing.Optional[typing.Any]
    :param anaglyph_type: Anaglyph Type
    :type anaglyph_type: typing.Optional[typing.Any]
    :param interlace_type: Interlace Type
    :type interlace_type: typing.Optional[typing.Any]
    :param use_interlace_swap: Swap Left/Right, Swap left and right stereo channels
    :type use_interlace_swap: typing.Union[bool, typing.Any]
    :param use_sidebyside_crosseyed: Cross-Eyed, Right eye should see left image and vice versa
    :type use_sidebyside_crosseyed: typing.Union[bool, typing.Any]
    '''

    pass


def splash(override_context: typing.Union[typing.
                                          Dict, 'bpy.types.Context'] = None,
           execution_context: typing.Union[str, int] = None,
           undo: typing.Optional[bool] = None):
    ''' Open the splash screen with release info

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def splash_about(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None):
    ''' Open a window with information about Blender

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def sysinfo(override_context: typing.Union[typing.
                                           Dict, 'bpy.types.Context'] = None,
            execution_context: typing.Union[str, int] = None,
            undo: typing.Optional[bool] = None,
            *,
            filepath: typing.Union[str, typing.Any] = ""):
    ''' Generate system information, saved into a text file

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: filepath
    :type filepath: typing.Union[str, typing.Any]
    '''

    pass


def tool_set_by_id(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None,
                   *,
                   name: typing.Union[str, typing.Any] = "",
                   cycle: typing.Union[bool, typing.Any] = False,
                   as_fallback: typing.Union[bool, typing.Any] = False,
                   space_type: typing.Optional[typing.Any] = 'EMPTY'):
    ''' Set the tool by name (for keymaps)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param name: Identifier, Identifier of the tool
    :type name: typing.Union[str, typing.Any]
    :param cycle: Cycle, Cycle through tools in this group
    :type cycle: typing.Union[bool, typing.Any]
    :param as_fallback: Set Fallback, Set the fallback tool instead of the primary tool
    :type as_fallback: typing.Union[bool, typing.Any]
    :param space_type: Type
    :type space_type: typing.Optional[typing.Any]
    '''

    pass


def tool_set_by_index(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None,
                      *,
                      index: typing.Optional[typing.Any] = 0,
                      cycle: typing.Union[bool, typing.Any] = False,
                      expand: typing.Union[bool, typing.Any] = True,
                      as_fallback: typing.Union[bool, typing.Any] = False,
                      space_type: typing.Optional[typing.Any] = 'EMPTY'):
    ''' Set the tool by index (for keymaps)

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param index: Index in Toolbar
    :type index: typing.Optional[typing.Any]
    :param cycle: Cycle, Cycle through tools in this group
    :type cycle: typing.Union[bool, typing.Any]
    :param expand: expand, Include tool subgroups
    :type expand: typing.Union[bool, typing.Any]
    :param as_fallback: Set Fallback, Set the fallback tool instead of the primary
    :type as_fallback: typing.Union[bool, typing.Any]
    :param space_type: Type
    :type space_type: typing.Optional[typing.Any]
    '''

    pass


def toolbar(override_context: typing.Union[typing.
                                           Dict, 'bpy.types.Context'] = None,
            execution_context: typing.Union[str, int] = None,
            undo: typing.Optional[bool] = None):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__. :file: `startup/bl_operators/wm.py\:1928 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/wm.py$1928>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def toolbar_fallback_pie(override_context: typing.
                         Union[typing.Dict, 'bpy.types.Context'] = None,
                         execution_context: typing.Union[str, int] = None,
                         undo: typing.Optional[bool] = None):
    ''' Undocumented, consider `contributing <https://developer.blender.org/T51061>`__. :file: `startup/bl_operators/wm.py\:1952 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/wm.py$1952>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def toolbar_prompt(override_context: typing.
                   Union[typing.Dict, 'bpy.types.Context'] = None,
                   execution_context: typing.Union[str, int] = None,
                   undo: typing.Optional[bool] = None):
    ''' Leader key like functionality for accessing tools :file: `startup/bl_operators/wm.py\:2052 <https://developer.blender.org/diffusion/B/browse/master/release/scripts/startup/bl_operators/wm.py$2052>`_

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def url_open(override_context: typing.Union[typing.
                                            Dict, 'bpy.types.Context'] = None,
             execution_context: typing.Union[str, int] = None,
             undo: typing.Optional[bool] = None,
             *,
             url: typing.Union[str, typing.Any] = ""):
    ''' Open a website in the web browser

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param url: URL, URL to open
    :type url: typing.Union[str, typing.Any]
    '''

    pass


def url_open_preset(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None,
                    *,
                    type: typing.Union[str, int, typing.Any] = '',
                    id: typing.Union[str, typing.Any] = ""):
    ''' Open a preset website in the web browser

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param type: Site
    :type type: typing.Union[str, int, typing.Any]
    :param id: Identifier, Optional identifier
    :type id: typing.Union[str, typing.Any]
    '''

    pass


def usd_export(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None,
               *,
               filepath: typing.Union[str, typing.Any] = "",
               check_existing: typing.Union[bool, typing.Any] = True,
               filter_blender: typing.Union[bool, typing.Any] = False,
               filter_backup: typing.Union[bool, typing.Any] = False,
               filter_image: typing.Union[bool, typing.Any] = False,
               filter_movie: typing.Union[bool, typing.Any] = False,
               filter_python: typing.Union[bool, typing.Any] = False,
               filter_font: typing.Union[bool, typing.Any] = False,
               filter_sound: typing.Union[bool, typing.Any] = False,
               filter_text: typing.Union[bool, typing.Any] = False,
               filter_archive: typing.Union[bool, typing.Any] = False,
               filter_btx: typing.Union[bool, typing.Any] = False,
               filter_collada: typing.Union[bool, typing.Any] = False,
               filter_alembic: typing.Union[bool, typing.Any] = False,
               filter_usd: typing.Union[bool, typing.Any] = True,
               filter_volume: typing.Union[bool, typing.Any] = False,
               filter_folder: typing.Union[bool, typing.Any] = True,
               filter_blenlib: typing.Union[bool, typing.Any] = False,
               filemode: typing.Optional[typing.Any] = 8,
               display_type: typing.Optional[typing.Any] = 'DEFAULT',
               sort_method: typing.Union[str, int, typing.Any] = '',
               selected_objects_only: typing.Union[bool, typing.Any] = False,
               visible_objects_only: typing.Union[bool, typing.Any] = True,
               export_animation: typing.Union[bool, typing.Any] = False,
               export_hair: typing.Union[bool, typing.Any] = False,
               export_uvmaps: typing.Union[bool, typing.Any] = True,
               export_normals: typing.Union[bool, typing.Any] = True,
               export_materials: typing.Union[bool, typing.Any] = True,
               use_instancing: typing.Union[bool, typing.Any] = False,
               evaluation_mode: typing.Optional[typing.Any] = 'RENDER'):
    ''' Export current scene in a USD archive

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    :param filepath: File Path, Path to file
    :type filepath: typing.Union[str, typing.Any]
    :param check_existing: Check Existing, Check and warn on overwriting existing files
    :type check_existing: typing.Union[bool, typing.Any]
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
    :param display_type: Display Type * ``DEFAULT`` Default, Automatically determine display type for files. * ``LIST_VERTICAL`` Short List, Display files as short list. * ``LIST_HORIZONTAL`` Long List, Display files as a detailed list. * ``THUMBNAIL`` Thumbnails, Display files as thumbnails.
    :type display_type: typing.Optional[typing.Any]
    :param sort_method: File sorting mode
    :type sort_method: typing.Union[str, int, typing.Any]
    :param selected_objects_only: Selection Only, Only selected objects are exported. Unselected parents of selected objects are exported as empty transform
    :type selected_objects_only: typing.Union[bool, typing.Any]
    :param visible_objects_only: Visible Only, Only visible objects are exported. Invisible parents of exported objects are exported as empty transform
    :type visible_objects_only: typing.Union[bool, typing.Any]
    :param export_animation: Animation, When checked, the render frame range is exported. When false, only the current frame is exported
    :type export_animation: typing.Union[bool, typing.Any]
    :param export_hair: Hair, When checked, hair is exported as USD curves
    :type export_hair: typing.Union[bool, typing.Any]
    :param export_uvmaps: UV Maps, When checked, all UV maps of exported meshes are included in the export
    :type export_uvmaps: typing.Union[bool, typing.Any]
    :param export_normals: Normals, When checked, normals of exported meshes are included in the export
    :type export_normals: typing.Union[bool, typing.Any]
    :param export_materials: Materials, When checked, the viewport settings of materials are exported as USD preview materials, and material assignments are exported as geometry subsets
    :type export_materials: typing.Union[bool, typing.Any]
    :param use_instancing: Instancing, When checked, instanced objects are exported as references in USD. When unchecked, instanced objects are exported as real objects
    :type use_instancing: typing.Union[bool, typing.Any]
    :param evaluation_mode: Use Settings for, Determines visibility of objects, modifier settings, and other areas where there are different settings for viewport and rendering * ``RENDER`` Render, Use Render settings for object visibility, modifier settings, etc. * ``VIEWPORT`` Viewport, Use Viewport settings for object visibility, modifier settings, etc.
    :type evaluation_mode: typing.Optional[typing.Any]
    '''

    pass


def window_close(override_context: typing.
                 Union[typing.Dict, 'bpy.types.Context'] = None,
                 execution_context: typing.Union[str, int] = None,
                 undo: typing.Optional[bool] = None):
    ''' Close the current window

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def window_fullscreen_toggle(override_context: typing.
                             Union[typing.Dict, 'bpy.types.Context'] = None,
                             execution_context: typing.Union[str, int] = None,
                             undo: typing.Optional[bool] = None):
    ''' Toggle the current window fullscreen

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def window_new(override_context: typing.
               Union[typing.Dict, 'bpy.types.Context'] = None,
               execution_context: typing.Union[str, int] = None,
               undo: typing.Optional[bool] = None):
    ''' Create a new window

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def window_new_main(override_context: typing.
                    Union[typing.Dict, 'bpy.types.Context'] = None,
                    execution_context: typing.Union[str, int] = None,
                    undo: typing.Optional[bool] = None):
    ''' Create a new main window with its own workspace and scene selection

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass


def xr_session_toggle(override_context: typing.
                      Union[typing.Dict, 'bpy.types.Context'] = None,
                      execution_context: typing.Union[str, int] = None,
                      undo: typing.Optional[bool] = None):
    ''' Open a view for use with virtual reality headsets, or close it if already opened

    :type override_context: typing.Union[typing.Dict, 'bpy.types.Context']
    :type execution_context: typing.Union[str, int]
    :type undo: typing.Optional[bool]
    '''

    pass
