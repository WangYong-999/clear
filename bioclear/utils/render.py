#!/bioclear/thirty/blender-2.93.3-linux-x64/2.93/python/bin/python3.9
import copy
import math
import os
import random
import shutil
import sys
#print(sys.version_info)
import time
import yaml
import bpy
from mathutils import Vector
import numpy as np 
from bpy_extras.object_utils import world_to_camera_view

RENDERING_PATH=os.path.dirname(os.path.abspath(__file__))
setting=yaml.load(open(os.path.join(RENDERING_PATH,"setting.yaml")),Loader=yaml.FullLoader)
SCENE_NUM=sys.argv[sys.argv.index("--")+1:][0] if len(sys.argv[sys.argv.index("--")+1:])>0 else 0
PATH_SCENE = os.path.join(setting["output_root_path"], str(SCENE_NUM).zfill(5))
if os.path.exists(setting["output_root_path"]):
    shutil.rmtree(setting["output_root_path"])
if not os.path.exists(setting["output_root_path"]):
    os.mkdir(setting["output_root_path"])
if not os.path.exists(PATH_SCENE):
    os.mkdir(PATH_SCENE)

g_shape_synset_name_pairs = copy.deepcopy(setting['g_shape_synset_name_pairs_all'])
g_shape_synset_name_pairs['00000000'] = 'other'
for item in setting['g_shape_synset_name_pairs_all']:
    if not setting['g_shape_synset_name_pairs_all'][item] in setting['selected_class']:
        g_shape_synset_name_pairs[item] = 'other'

class Material():
    def __init__(self) -> None:
        pass
    def modifyMaterial(self,mat_links, mat_nodes, material_name, is_texture=False, orign_base_color=None, tex_node=None):
        # 随机混合率
        tex_mix_prop=random.uniform(0.7,0.95)
        mix_prop=random.uniform(00.6,0.9)
        # 随机:对于一种材质，保留/不保留物体本身的纹理或颜色= 3:1
        transfer_rand=random.randint(0,3)
        if transfer_rand==0:
            transfer_flag=False
        else:
            transfer_flag=True
        # 随机:原则BSDF，一种着色器
        bsdfnode_list=[n for n in mat_nodes if isinstance(n,bpy.types.ShaderNodeBsdfPrincipled)]
        if bsdfnode_list!=[]:
            for bsdfnode in bsdfnode_list:
                if bsdfnode.inputs[4].links:#metallic
                    src_value=bsdfnode.inputs[4].default_value
                    if material_name.split("_")[0]=="metal":
                        new_value=src_value+random.uniform(-0.05,0.05)
                    elif material_name.split("_")[0]=="porcelain":#瓷
                        new_value=src_value+random.uniform(-0.05,0.1)
                    elif material_name.split("_")[0] == "plasticsp":#
                        new_value = src_value + random.uniform(-0.05, 0.1)
                    else:
                        new_value = src_value + random.uniform(-0.05, 0.05)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[4].default_value = new_value

                if not bsdfnode.inputs[5].links:    # specular
                    src_value = bsdfnode.inputs[5].default_value
                    #if material_name.split("_")[0] == "metal":
                    new_value = src_value + random.uniform(0, 0.3)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[5].default_value = new_value
                if not bsdfnode.inputs[6].links:    # specularTint
                    src_value = bsdfnode.inputs[6].default_value
                    new_value = src_value + random.uniform(-1, 1)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[6].default_value = new_value
                if not bsdfnode.inputs[7].links:    # roughness
                    src_value = bsdfnode.inputs[7].default_value
                    if material_name.split("_")[0] == "metal" or material_name.split("_")[0] == "porcelain" or material_name.split("_")[0] == "plasticsp" or material_name.split("_")[0] == "paintsp":
                        new_value = src_value + random.uniform(-0.1, 0.03)
                    else:
                        new_value = src_value + random.uniform(-0.03, 0.1)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[7].default_value = new_value
                if not bsdfnode.inputs[8].links:    # anisotropic
                    src_value = bsdfnode.inputs[8].default_value
                    new_value = src_value + random.uniform(-0.1, 0.1)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[8].default_value = new_value
                if not bsdfnode.inputs[9].links:    # anisotropicRotation
                    src_value = bsdfnode.inputs[9].default_value
                    new_value = src_value + random.uniform(-0.3, 0.3)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[9].default_value = new_value
                if not bsdfnode.inputs[10].links:    # sheen
                    src_value = bsdfnode.inputs[10].default_value
                    new_value = src_value + random.uniform(-0.1, 0.1)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[10].default_value = new_value
                if not bsdfnode.inputs[11].links:    # sheenTint
                    src_value = bsdfnode.inputs[11].default_value
                    new_value = src_value + random.uniform(-0.2, 0.2)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[11].default_value = new_value
                if not bsdfnode.inputs[12].links:    # clearcoat
                    src_value = bsdfnode.inputs[12].default_value
                    new_value = src_value + random.uniform(-0.2, 0.2)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[12].default_value = new_value
                if not bsdfnode.inputs[13].links:    # clearcoatGloss
                    src_value = bsdfnode.inputs[13].default_value
                    new_value = src_value + random.uniform(-0.2, 0.2)
                    if new_value > 1.0: new_value = 1.0
                    elif new_value < 0: new_value = 0.0
                    bsdfnode.inputs[13].default_value = new_value

        ## metal
        if material_name == "metal_0":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.3, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 1.0)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.3, 1.0)         # clearcoatGloss

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
                mat_links.new(mat_nodes["Image Texture.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_1":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.9, 1.00)        # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.08, 0.25)         # roughness
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.04, 0.5)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.3, 0.7)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.8, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22]) 

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_10":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.5)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.3, 0.7)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                bsdf_new.location = Vector((-800, 0))
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'
                mix_new.location = Vector((-800, 0))

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7

                mat_links.new(mat_nodes["Image Texture"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[19])
                mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[4])
                mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
                mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_11":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.8)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[4])    
                mat_links.new(mat_nodes["Image Texture.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])  
                mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])                  
        elif material_name == "metal_12":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.8)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])    
                mat_links.new(mat_nodes["Reroute.006"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_13":
            ## Random Parameters
            # mat_nodes["Principled BSDF.001"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF.001"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF.001"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF.001"].inputs[8].default_value = random.uniform(0.3, 0.7)         # anisotropic
            # mat_nodes["Principled BSDF.001"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
            # mat_nodes["Principled BSDF.001"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF.001"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF.001"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20]) 
                mat_links.new(mat_nodes["Mix.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])  

                mat_links.new(mat_nodes["Principled BSDF.001"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output.001"].inputs["Surface"])  
        elif material_name == "metal_14":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.5)         # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.5)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.85 
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7 
                
                mat_links.new(mat_nodes["Group"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[7])  
                mat_links.new(mat_nodes["Group"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])   

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_2":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.95)        # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)        # clearcoatGloss

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Image Texture.003"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
                mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])       
        elif material_name == "metal_3":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)        # clearcoatGloss
            mat_nodes["Gamma"].inputs[1].default_value = random.uniform(3.0, 4.0)

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Gamma"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
                mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])       
        elif material_name == "metal_4":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.1, 0.5)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.5)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.5)        # clearcoatGloss
    
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
                mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])  
        elif material_name == "metal_5":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.2, 0.4)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.6, 0.9)        # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.8, 1.0)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

                mat_links.new(mat_nodes["Voronoi Texture"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[7]) 
                mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22])   
                mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[21])  

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])  
        elif material_name == "metal_6":
            ## Random Parameters
            # mat_nodes["BSDF guidé"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
            # mat_nodes["BSDF guidé"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["BSDF guidé"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["BSDF guidé"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
            # mat_nodes["BSDF guidé"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["BSDF guidé"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
            # mat_nodes["BSDF guidé"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
            mat_nodes["Valeur"].outputs[0].default_value = random.uniform(0.1, 0.3)

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                bsdf_new.location = Vector((-800, 0))
                for key, input in enumerate(mat_nodes["BSDF guidé"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'
                mix_new.location = Vector((-800, 0))

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7 

                mat_links.new(mat_nodes["Mélanger.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

                mat_links.new(mat_nodes["BSDF guidé"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Sortie de matériau"].inputs[0])
        elif material_name == "metal_7":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.7, 0.9)        # anisotropic
            # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
            
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                #bsdf_new.location = Vector((-800, 0))
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'
                #mix_new.location = Vector((-800, 0))

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9 
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7

                mat_links.new(mat_nodes["Reroute.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])
                mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
                mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22])

                mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "metal_8":
            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value
                bsdf_1_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_1_new.name = 'Principled BSDF-1-new'
                for key, input in enumerate(mat_nodes["Principled BSDF.001"].inputs):
                    bsdf_1_new.inputs[key].default_value = input.default_value
                bsdf_2_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_2_new.name = 'Principled BSDF-2-new'
                for key, input in enumerate(mat_nodes["Principled BSDF.002"].inputs):
                    bsdf_2_new.inputs[key].default_value = input.default_value
                bsdf_3_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_3_new.name = 'Principled BSDF-3-new'
                for key, input in enumerate(mat_nodes["Principled BSDF.003"].inputs):
                    bsdf_3_new.inputs[key].default_value = input.default_value
                
                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'
                mix_1_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_1_new.name = 'Mix Shader-1-new'
                mix_2_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_2_new.name = 'Mix Shader-2-new'
                mix_3_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_3_new.name = 'Mix Shader-3-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs[0])
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-1-new"].inputs[0])
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-2-new"].inputs[0])
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-3-new"].inputs[0])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.6
                    mat_nodes["Mix Shader-1-new"].inputs[0].default_value = 0.6
                    mat_nodes["Mix Shader-2-new"].inputs[0].default_value = 0.6
                    mat_nodes["Mix Shader-3-new"].inputs[0].default_value = 0.6
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Principled BSDF-1-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Principled BSDF-2-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Principled BSDF-3-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.5
                    mat_nodes["Mix Shader-1-new"].inputs[0].default_value = 0.5
                    mat_nodes["Mix Shader-2-new"].inputs[0].default_value = 0.5
                    mat_nodes["Mix Shader-3-new"].inputs[0].default_value = 0.5

                mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])
                mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-1-new"].inputs[20]) 
                mat_links.new(mat_nodes["Bump.001"].outputs[0], mat_nodes["Principled BSDF-2-new"].inputs[20])   

                mat_links.new(mat_nodes["Principled BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])

                mat_links.new(mat_nodes["Principled BSDF.001"].outputs[0], mat_nodes["Mix Shader-1-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-1-new"].outputs[0], mat_nodes["Mix Shader-1-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-1-new"].outputs[0], mat_nodes["Mix Shader"].inputs[2])

                mat_links.new(mat_nodes["Principled BSDF.002"].outputs[0], mat_nodes["Mix Shader-2-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-2-new"].outputs[0], mat_nodes["Mix Shader-2-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-2-new"].outputs[0], mat_nodes["Mix Shader.001"].inputs[1])

                mat_links.new(mat_nodes["Principled BSDF.003"].outputs[0], mat_nodes["Mix Shader-3-new"].inputs[1])
                mat_links.new(mat_nodes["Principled BSDF-3-new"].outputs[0], mat_nodes["Mix Shader-3-new"].inputs[2])
                mat_links.new(mat_nodes["Mix Shader-3-new"].outputs[0], mat_nodes["Mix Shader.001"].inputs[2])        
        elif material_name == "metal_9":
            ## Random Parameters
            # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
            # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
            # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
            mat_nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.01, 0.3)         # roughness
            # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
            # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
            mat_nodes["Anisotropic BSDF"].inputs[1].default_value = random.uniform(0.11, 0.25)
            mat_nodes["Anisotropic BSDF"].inputs[2].default_value = random.uniform(0.4, 0.6)

            if transfer_flag == True:
                bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_new.name = 'Principled BSDF-new'
                for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                    bsdf_new.inputs[key].default_value = input.default_value

                mix_new = mat_nodes.new(type='ShaderNodeMixShader')
                mix_new.name = 'Mix Shader-new'

                if is_texture:
                    mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9
                    mat_links.new(tex_node.outputs[0], mat_nodes["Anisotropic BSDF"].inputs[0])
                else:
                    mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                    mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9
                    mat_nodes["Anisotropic BSDF"].inputs[0].default_value = list(orign_base_color)

                mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
                mat_links.new(mat_nodes["Principled BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
                mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])

        ## porcelain
        elif material_name == "porcelain_0":
            # if is_texture:
            #     mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            # else:
            #     mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9   
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.8   

            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])    
        elif material_name == "porcelain_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
            else:
                mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color)  
        elif material_name == "porcelain_2":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9   
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.8   

            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])    
        elif material_name == "porcelain_3":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix.001"].inputs[1])
            else:
                mat_nodes["Mix.001"].inputs[1].default_value = list(orign_base_color)   
        elif material_name == "porcelain_4":
            # if is_texture:
            #     mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
            #     mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
            # else:
            #     mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
            #     mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.05, 0.15)

            diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
            diff_new.name = 'Diffuse BSDF-new'
            for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
                diff_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
            else:
                mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

            mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
        elif material_name == "porcelain_5":
            # if is_texture:
            #     mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
            #     mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
            # else:
            #     mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
            #     mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
            diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
            diff_new.name = 'Diffuse BSDF-new'
            for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
                diff_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
            else:
                mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

            mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
        elif material_name == "porcelain_6":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
            else:
                mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)  
        
        ## plastic
        elif material_name == "plastic_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
            else:
                mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)            
        elif material_name == "plastic_2":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
            else:
                mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)    
        elif material_name == "plastic_3":
            mat_nodes["值(明度)"].outputs[0].default_value = random.uniform(0.05, 0.25)

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
            else:
                mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)  
        elif material_name == "plastic_5":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)    
        elif material_name == "plastic_6":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.012"].inputs[0])    
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.021"].inputs[0])    
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.022"].inputs[0])    
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.033"].inputs[0])  
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
                mat_nodes["RGB.001"].outputs[0].default_value = list(orign_base_color)
                """
                mat_nodes["RGB.002"].outputs[0].default_value = list(orign_base_color)
                mat_nodes["RGB.003"].outputs[0].default_value = list(orign_base_color)
                """

        ## rubber
        elif material_name == "rubber_0":
            diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
            diff_new.name = 'Diffuse BSDF-new'
            for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
                diff_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
            else:
                mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

            mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
        elif material_name == "rubber_1":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["RGB Curves"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "rubber_2":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["RGB Curves"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "rubber_3":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "rubber_4":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        
        ## plastic_specular
        elif material_name == "plasticsp_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.001"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute"].inputs[0])
            else:
                mat_nodes["RGB.001"].outputs[0].default_value = list(orign_base_color)
        elif material_name == "plasticsp_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        
        ## paint_specular
        elif material_name == "paintsp_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
        elif material_name == "paintsp_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[2])
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix.001"].inputs[1])
                mat_links.new(tex_node.outputs[0], mat_nodes["Hue Saturation Value"].inputs[4])
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
        elif material_name == "paintsp_2":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Group"].inputs[0])
            else:
                mat_nodes["Group"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "paintsp_3":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "paintsp_4":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF.001"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Glossy BSDF.001"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "paintsp_5":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Invert"].inputs[1])
                mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.002"].inputs[0])
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)

        ## rubber
        elif material_name == "rubber_5":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Mix.005"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])

        ## plastic
        elif material_name == "plastic_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
            else:
                mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_4":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "plastic_7":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])    
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_8":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Group"].inputs[0])    
            else:
                mat_nodes["Group"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_9":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)            
        elif material_name == "plastic_10":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)            
        elif material_name == "plastic_11":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_12":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[2])    
            else:
                mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_13":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "plastic_14":
            mat_nodes["Math.005"].inputs[1].default_value = random.uniform(0.05, 0.3)

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        
        ## paper
        elif material_name == "paper_0":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Mix.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "paper_1":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.8, 0.95)  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.8, 0.9)  

            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        elif material_name == "paper_2":
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.9, 0.95)  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.9, 0.95)  

            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Bright/Contrast"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
        
        ## leather
        elif material_name == "leather_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "leather_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "leather_2":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
            else:
                mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color)
        elif material_name == "leather_3":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "leather_4":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        elif material_name == "leather_5":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["lether"].inputs[0])
            else:
                mat_nodes["lether"].inputs[0].default_value = list(orign_base_color)

        ## wood
        elif material_name == "wood_0":
            pass
        elif material_name == "wood_1":
            pass
        elif material_name == "wood_2":
            pass
        elif material_name == "wood_3":
            pass
        elif material_name == "wood_4":
            pass
        elif material_name == "wood_5":
            pass
        elif material_name == "wood_6":
            pass
        elif material_name == "wood_7":
            pass
        elif material_name == "wood_8":
            pass
        elif material_name == "wood_9":
            pass

        ## fabric
        elif material_name == "fabric_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "fabric_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "fabric_2":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
            else:
                mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 

        ## clay
        elif material_name == "clay_0":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "clay_1":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "clay_2":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "clay_3":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
            else:
                mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 
        elif material_name == "clay_4":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
        elif material_name == "clay_5":
            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
            else:
                mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 

        ## glass
        elif material_name == "glass_0":
            mat_nodes["Mix Shader"].inputs[0].default_value = random.uniform(0.1, 0.3)
            mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.1, 0.3)
        elif material_name == "glass_4":
            mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.3, 0.7)
            mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.05, 0.2)
        elif material_name == "glass_5":
            mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.2, 0.4)
            mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.0, 0.1)
        elif material_name == "glass_14":
            mat_nodes["Glass BSDF.005"].inputs[1].default_value = random.uniform(0.0, 0.1)
            mat_nodes["Glass BSDF.006"].inputs[1].default_value = random.uniform(0.0, 0.1)
            mat_nodes["Glass BSDF.007"].inputs[1].default_value = random.uniform(0.0, 0.1)
            mat_nodes["Glass BSDF.008"].inputs[1].default_value = random.uniform(0.0, 0.1)
            mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.81, 0.87)
            mat_nodes["Layer Weight.001"].inputs[0].default_value = random.uniform(0.65, 0.71)
            mat_nodes["Layer Weight.002"].inputs[0].default_value = random.uniform(0.81, 0.87)
            color_value = random.uniform(0.599459, 0.70)
            mat_nodes["Transparent BSDF"].inputs[0].default_value = list([color_value, color_value, color_value, 1])

        # elif material_name == "glass_15":
        #     mat_nodes["Glass BSDF"].inputs[1].default_value = random.uniform(0.0, 0.1)
        #     mat_nodes["Glass BSDF"].inputs[2].default_value = random.uniform(1.325, 1.335)
        #     color_value = random.uniform(0.297, 0.35)
        #     mat_nodes["Transparent BSDF"].inputs[0].default_value = list([color_value, color_value, color_value, 1])

        else:
            print(material_name + " no change")

    def setModifyMaterial(self,obj,material):
        for mat_slot in obj.material_slots:
            #print("-material_slots:",mat_slot)
            if mat_slot.material:
                if mat_slot.material.node_tree:
                    #print("--material:" + str(mat_slot.material.name))
                    # Copy one material from the library
                    srcmat = material
                    #print(srcmat.name)
                    mat = srcmat.copy()
                    mat.name = mat_slot.material.name   # rename
                    mat_links = mat.node_tree.links
                    mat_nodes = mat.node_tree.nodes
                    # Obtain the Principled BSDF node
                    bsdf_node = mat_slot.material.node_tree.nodes.get("Principled BSDF", None)
                    if bsdf_node is not None:
                        # Obtain the Texture node, and determine w or w/o texture map
                        tex_node_orign = mat_slot.material.node_tree.nodes.get("Image Texture", None)
                        # if the texture map exists
                        if tex_node_orign is not None:
                            #mat = mat_slot.material.copy() 
                            # Get the bl_idname to create a new node of the same type
                            tex_node = mat_nodes.new(tex_node_orign.bl_idname)
                            texture_img = bpy.data.images[tex_node_orign.image.name]
                            # Assign the default values from the old node to the new node
                            tex_node.image = texture_img
                            #tex_node.location = Vector((-800, 0))
                            # create the new link for mat
                            self.modifyMaterial(mat_links, mat_nodes, srcmat.name, is_texture=True, tex_node=tex_node)
                        # if the texture map does not exist
                        else:
                            # obtain the base color
                            orign_base_color = mat_slot.material.node_tree.nodes["Principled BSDF"].inputs[0].default_value
                            if orign_base_color[0] == 0.0 and orign_base_color[1] == 0.0 and orign_base_color[2] == 0.0:
                                orign_base_color == [0.05, 0.05, 0.05, 1]
                            # apply to mat
                            self.modifyMaterial(mat_links, mat_nodes, srcmat.name, is_texture=False, orign_base_color=orign_base_color)

                    # apply the material
                    bpy.data.materials.remove(mat_slot.material)
                    mat_slot.material = mat

    def setModifyRawMaterial(self,obj):
        for mat_slot in obj.material_slots:
          if mat_slot.material:
            if mat_slot.material.node_tree:
                bsdf_node = mat_slot.material.node_tree.nodes.get("Principled BSDF", None)
                if bsdf_node is not None:
                    # Obtain the Texture node, and determine w or w/o texture map
                    tex_node_orign = mat_slot.material.node_tree.nodes.get("Image Texture", None)
                    # if the texture map does not exist
                    if tex_node_orign is None:
                            # obtain base color
                            orign_base_color = mat_slot.material.node_tree.nodes["Principled BSDF"].inputs[0].default_value
                            if orign_base_color[0] == 0.0 and orign_base_color[1] == 0.0 and orign_base_color[2] == 0.0:
                                mat = mat_slot.material.copy()
                                mat.name = mat_slot.material.name   # rename
                                mat_nodes = mat.node_tree.nodes   
                                mat_nodes["Principled BSDF"].inputs[0].default_value = list([0.05, 0.05, 0.05, 1])
                                # apply the material
                                bpy.data.materials.remove(mat_slot.material)
                                mat_slot.material = mat

class BlenderRender():
    def __init__(self,viewport_size_x,viewport_size_y) -> None:
        '''移除相机灯光，初始化相机灯光、渲染器、节点管理器'''
        # 1、移除场景中的所有物体相机和灯光
        for obj in bpy.data.meshes:
            bpy.data.meshes.remove(obj)
        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj,do_unlink=True)
        # 2、初始化所需要的设置
        # 2.1 左右相机，相机也是一个Object
        camera_l_data = bpy.data.cameras.new(name="camera_l")
        camera_l_object=bpy.data.objects.new(name="camera_l",object_data=camera_l_data)
        bpy.context.collection.objects.link(camera_l_object)

        camera_r_data = bpy.data.cameras.new(name="camera_r")
        camera_r_object=bpy.data.objects.new(name="camera_r",object_data=camera_r_data)
        bpy.context.collection.objects.link(camera_r_object)

        camera_l = bpy.data.objects["camera_l"]
        camera_r = bpy.data.objects["camera_r"]
        
        camera_l.location=(1,0,0)
        camera_r.location=(1,0,0)

        # 2.2 光源
        light_emitter_data = bpy.data.lights.new(name="light_emitter",type="SPOT")
        light_emitter_object=bpy.data.objects.new(name="light_emitter",object_data=light_emitter_data)
        bpy.context.collection.objects.link(light_emitter_object)

        light_emitter=bpy.data.objects["light_emitter"]
        light_emitter.location=(1,0,0)
        light_emitter.data.energy=setting["LIGHT_EMITTER_ENERGY"]

        # 2.3 渲染器
        render_context=bpy.context.scene.render
        render_context.resolution_percentage=100

        self.render_context=render_context 
        self.camera_l=camera_l
        self.camera_r=camera_r
        self.light_emitter=light_emitter
        self.model_loaded=False#加载模型之后未True
        self.background_added=None
        
        self.render_context.resolution_x=viewport_size_x
        self.render_context.resolution_y=viewport_size_y

        self.render_context.image_settings.file_format='PNG'# 输出
        self.render_context.image_settings.compression=0
        self.render_context.image_settings.color_mode='BW'
        self.render_context.image_settings.color_depth='8'
        self.render_context.engine='CYCLES'#渲染器
        
        bpy.context.scene.cycles.progressive='BRANCHED_PATH'
        bpy.context.scene.cycles.use_denoising=True
        bpy.context.scene.cycles.denoiser='NLM'
        bpy.context.scene.cycles.film_exposure=0.5

        bpy.context.scene.view_layers["View Layer"].use_sky=True

        # 2.3 获取节点管理器
        bpy.context.scene.use_nodes=True
        tree=bpy.context.scene.node_tree
        links=tree.links

        for n in tree.nodes:
            tree.nodes.remove(n)
        
        rl=tree.nodes.new('CompositorNodeRLayers')#输入

        # 深度图
        self.fileOutput=tree.nodes.new(type="CompositorNodeOutputFile")#输出
        self.fileOutput.base_path="./new_data/0000"
        self.fileOutput.format.file_format='OPEN_EXR'#输出格式
        self.fileOutput.format.color_depth='32'
        self.fileOutput.file_slots[0].path='depth'
        # for i in rl.outputs:
        #     print(i)
        #     print('\n')
        # exit()
        links.new(rl.outputs['Depth'],self.fileOutput.inputs[0])#连接
    
        # new yong
        # 法线图
        scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        scale_normal.blend_type = 'MULTIPLY'
        scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(rl.outputs['Normal'], scale_normal.inputs[1])
        bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        bias_normal.blend_type = 'ADD'
        bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_normal.outputs[0], bias_normal.inputs[1])
        self.normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        self.normal_file_output.label = 'Normal Output'
        self.normal_file_output.format.file_format='OPEN_EXR'
        self.normal_file_output.format.color_depth='32'
        links.new(bias_normal.outputs[0], self.normal_file_output.inputs[0])

        # 掩码图
        scale_normal = tree.nodes.new(type="CompositorNodeIDMask")
        # scale_normal.blend_type = 'MULTIPLY'
        # scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        # links.new(rl.outputs['Normal'], scale_normal.inputs[1])
        # bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        # bias_normal.blend_type = 'ADD'
        # bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        # links.new(scale_normal.outputs[0], bias_normal.inputs[1])
        # self.normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        # self.normal_file_output.label = 'Normal Output'
        # self.normal_file_output.format.file_format='OPEN_EXR'
        # self.normal_file_output.format.color_depth='32'
        # links.new(bias_normal.outputs[0], self.normal_file_output.inputs[0])


        self.my_material={}
        self.render_mode='IR'#渲染模式

        self.pattern=[]#深度图片
        self.env_map=[]#环境背景

        # 3 设置Blender环境属性
        self.set(setting["env_map_path"],setting["background_size"],setting["background_position"],setting["background_scale"],setting["max_instance_num"])

        # 4 初始化Material
        self.material=Material()
    
    def set(self,env_map_path,background_size,background_position,background_scale,mask_num):
        '''设置环境属性：背景、材质'''
        # 1、加载背景、材料
        for img in bpy.data.images:
            if img.filepath.split("/")[-1]=="pattern.png":
                self.pattern=img
                break
        for item in os.listdir(env_map_path):
            if item.split('.')[-1]=='hdr':
                self.env_map.append(bpy.data.images.load(filepath=os.path.join(env_map_path,item)))

        # 2、设置环境(着色器)节点管理器
        node_tree=bpy.context.scene.world.node_tree
        # bpy.context.scene.node_tree
        tree_nodes=node_tree.nodes

        # 2.1清除所有节点
        tree_nodes.clear()
        # 2.2 增加背景节点、增加材质节点
        node_background=tree_nodes.new(type='ShaderNodeBackground')
        node_tex_enviroment=tree_nodes.new(type='ShaderNodeTexEnvironment')
        node_tex_enviroment.location=-300,0

        node_tex_coord=tree_nodes.new(type='ShaderNodeTexCoord')
        node_tex_coord.location=-700,0

        node_mapping=tree_nodes.new('ShaderNodeMapping')
        node_mapping.location=-500,0

        node_output=tree_nodes.new(type='ShaderNodeOutputWorld')
        node_output.location=200,0

        # 2.3 连接
        links=node_tree.links
        links.new(node_tex_enviroment.outputs["Color"],node_background.inputs["Color"])
        links.new(node_background.outputs["Background"],node_output.inputs["Surface"])
        links.new(node_tex_coord.outputs["Generated"],node_mapping.inputs["Vector"])
        links.new(node_mapping.outputs["Vector"],node_tex_enviroment.inputs["Vector"])

        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value=1.0

        # 3 设置 背景
        # 3.1 定义背景材料
        material_name="default_background"
        material_background=(bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

        material_background.use_nodes=True
        material_background.node_tree.nodes.clear()
        node_tree=material_background.node_tree

        node_1=node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2=node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        node_3=node_tree.nodes.new('ShaderNodeTexImage')

        links=node_tree.links
        links.new(node_3.outputs[0],node_2.inputs[0])
        links.new(node_2.outputs[0],node_1.inputs[0])
        
        node_3.image=bpy.data.images.load(filepath=os.path.join(setting["working_root"],"texture/texture_0.jpg"))
        self.my_material["default_background"]=material_background

        # 3.2 增加背景object
        for i in range(-2,3,1):#增加背景平面
            for j in range(-2,3,1):
                position_i_j=(i*background_size+background_position[0],j*background_size+background_position[1],background_position[2])
                bpy.ops.mesh.primitive_plane_add(size=background_size,enter_editmode=False,align='WORLD',location=position_i_j,scale=background_scale)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type='PASSIVE'
                bpy.context.object.rigid_body.collision_shape='BOX'
        
        for i in range(-2,3,1):
            for j in [-2,2]:
                position_i_j=(i*background_size+background_position[0],j*background_size+background_position[1],background_position[2]-0.25)
                rotation_elur=(math.pi/2.0,0.0,0.0)
                bpy.ops.mesh.primitive_plane_add(size=background_size,enter_editmode=False,align='WORLD',location=position_i_j,rotation=rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type='PASSIVE'
                bpy.context.object.rigid_body.collision_shape='BOX'

        for i in range(-2,3,1):
            for j in [-2,2]:
                position_i_j=(i*background_size+background_position[0],j*background_size+background_position[1],background_position[2]-0.25)
                rotation_elur=(math.pi/2.0,0.0,0.0)
                bpy.ops.mesh.primitive_plane_add(size=background_size,enter_editmode=False,align='WORLD',location=position_i_j,rotation=rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type='PASSIVE'
                bpy.context.object.rigid_body.collision_shape='BOX'

        # 3.3 将背景材料附加到背景object
        count=0
        for obj in bpy.data.objects:#
            if obj.type=="MESH":
                obj.name="background_"+str(count)
                obj.data.name="background_"+str(count)
                obj.active_material=material_background
                count+=1
        self.background_added=True

        # 4 增加材质库
        mat_specular_list=[]#镜面
        mat_transparent_list=[]#透明
        mat_diffuse_list=[]#扩散
        mat_background_list=[]#背景
        for mat in bpy.data.materials:
            name_class=mat.name.split('_')[0]
            if name_class in setting["material_class_instance_pairs"]['specular']:
                mat_specular_list.append(mat)
            if name_class in setting["material_class_instance_pairs"]['transparent']:
                mat_transparent_list.append(mat)
            if name_class in setting["material_class_instance_pairs"]['diffuse']:
                mat_diffuse_list.append(mat)
            if name_class in setting["material_class_instance_pairs"]['background']:
                mat_background_list.append(mat)
        
        self.my_material['specular']=mat_specular_list
        self.my_material['transparent']=mat_transparent_list
        self.my_material['diffuse']=mat_diffuse_list
        self.my_material['background']=mat_background_list
        
        # 5 增加掩码材质
        material_name="mask_background"
        material_mask=(bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

        material_mask.use_nodes=True
        node_tree=material_mask.node_tree
        material_mask.node_tree.nodes.clear()

        node_1=node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2=node_tree.nodes.new('ShaderNodeBrightContrast')

        links=node_tree.links
        # links.new(node_2.outputs[0],node_1.inputs[0])
        links.new(node_1.inputs[0],node_2.outputs[0])
        node_2.inputs[0].default_value=(1,1,1,1)

        self.my_material[material_name]=material_mask

        for i in range(mask_num):
            class_name=str(i+1)
            material_name="mask_"+class_name
            material_mask=(bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

            material_mask.use_nodes=True
            node_tree=material_mask.node_tree
            material_mask.node_tree.nodes.clear()

            node_1=node_tree.nodes.new('ShaderNodeOutputMaterial')
            node_2=node_tree.nodes.new('ShaderNodeBrightContrast')

            links=node_tree.links
            # links.new(node_2.outputs[0],node_1.inputs[0])
            links.new(node_1.inputs[0],node_2.outputs[0])

            if class_name.split('_')[0]=='background':
                node_2.inputs[0].default_value=(1,1,1,1)
            else:
                node_2.inputs[0].default_value=((i+1)/255.,0.,0.,1)
            
            self.my_material[material_name]=material_mask
            
        # 6 增加NOCS材质
        material_name='coord_color'
        mat=(bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))
            
        mat.use_nodes=True
        mat.node_tree.nodes.clear()
        node_tree=mat.node_tree
        node_tree.links.clear()

        node_R=node_tree.nodes.new('ShaderNodeVertexColor')
        node_G=node_tree.nodes.new('ShaderNodeVertexColor')
        node_B=node_tree.nodes.new('ShaderNodeVertexColor')
        node_R.layer_name="Col_R"
        node_G.layer_name="Col_G"
        node_B.layer_name="Col_B"

        node_Output=node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_Emission=node_tree.nodes.new('ShaderNodeEmission')
        node_LightPath=node_tree.nodes.new('ShaderNodeLightPath')
        node_Mix=node_tree.nodes.new('ShaderNodeMixShader')
        node_Combine=node_tree.nodes.new('ShaderNodeCombineRGB')

        node_tree.links.new(node_R.outputs[1],node_Combine.inputs[0])
        node_tree.links.new(node_G.outputs[1],node_Combine.inputs[1])
        node_tree.links.new(node_B.outputs[1],node_Combine.inputs[2])
        node_tree.links.new(node_Combine.outputs[0],node_Emission.inputs[0])
        node_tree.links.new(node_LightPath.outputs[0],node_Mix.inputs[0])
        node_tree.links.new(node_Emission.outputs[0],node_Mix.inputs[2])
        node_tree.links.new(node_Mix.outputs[0],node_Output.inputs[0])

        self.my_material[material_name]=mat

        # 7 增加法线材质
        material_name='normal'
        mat=(bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))
            
        mat.use_nodes=True
        node_tree=mat.node_tree
        nodes=node_tree.nodes
        nodes.clear()
        links=node_tree.links
        links.clear()

        node_new=nodes.new('ShaderNodeMath')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location=(151.59744262695312, 854.5482177734375)
        node_new.name='Math'
        node_new.operation='MULTIPLY'
        node_new.select=False
        node_new.use_clamp=False
        node_new.width=140.0
        node_new.inputs[0].default_value=0.5
        node_new.inputs[1].default_value=1.0
        node_new.inputs[2].default_value=0.0
        node_new.outputs[0].default_value=0.0

        node_new=nodes.new('ShaderNodeLightPath')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location=(602.9912719726562, 1046.660888671875)
        node_new.name='Light Path'
        node_new.select=False
        node_new.width=140.0
        for i in range(13):
            node_new.outputs[i].default_value=0.0
        
        node_new=nodes.new('ShaderNodeOutputMaterial')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.is_active_output = True
        node_new.location=(1168.93017578125, 701.84033203125)
        node_new.name='Material Output'
        node_new.select=False
        node_new.target='ALL'
        node_new.width=140.0
        node_new.inputs[2].default_value=[0.0,0.0,0.0]

        node_new=nodes.new('ShaderNodeBsdfTransparent')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location=(731.72900390625, 721.4832763671875)
        node_new.name='Transparent BSDF'
        node_new.select=False
        node_new.width=140.0
        node_new.inputs[0].default_value=[1.0,1.0,1.0,1.0]

        node_new=nodes.new('ShaderNodeCombineXYZ')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location=(594.4229736328125, 602.9271240234375)
        node_new.name='Combine XYZ'
        node_new.select=False
        node_new.width=140.0
        node_new.inputs[0].default_value=0.0
        node_new.inputs[1].default_value=0.0
        node_new.inputs[2].default_value=0.0
        node_new.outputs[0].default_value=[0.0,0.0,0.0]

        node_new=nodes.new('ShaderNodeMixShader')
        node_new.active_preview=False
        node_new.color=(0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location=(992.7239990234375, 707.2142333984375)
        node_new.name='Mix Shader'
        node_new.select=False
        node_new.width=140.0
        node_new.inputs[0].default_value=0.5
        
        node_new= nodes.new(type='ShaderNodeEmission')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location = (774.0802612304688, 608.2547607421875)
        node_new.name = 'Emission'
        node_new.select = False
        node_new.width = 140.0
        node_new.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]
        node_new.inputs[1].default_value = 1.0

        node_new = nodes.new(type='ShaderNodeSeparateXYZ')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location = (-130.12167358398438, 558.1497802734375)
        node_new.name = 'Separate XYZ'
        node_new.select = False
        node_new.width = 140.0
        node_new.inputs[0].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[0].default_value = 0.0
        node_new.outputs[1].default_value = 0.0
        node_new.outputs[2].default_value = 0.0

        node_new = nodes.new(type='ShaderNodeMath')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location = (162.43240356445312, 618.8094482421875)
        node_new.name = 'Math.002'
        node_new.operation = 'MULTIPLY'
        node_new.select = False
        node_new.use_clamp = False
        node_new.width = 140.0
        node_new.inputs[0].default_value = 0.5
        node_new.inputs[1].default_value = 1.0
        node_new.inputs[2].default_value = 0.0
        node_new.outputs[0].default_value = 0.0

        node_new = nodes.new(type='ShaderNodeMath')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location = (126.8158187866211, 364.5539855957031)
        node_new.name = 'Math.001'
        node_new.operation = 'MULTIPLY'
        node_new.select = False
        node_new.use_clamp = False
        node_new.width = 140.0
        node_new.inputs[0].default_value = 0.5
        node_new.inputs[1].default_value = -1.0
        node_new.inputs[2].default_value = 0.0
        node_new.outputs[0].default_value = 0.0

        node_new = nodes.new(type='ShaderNodeVectorTransform')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.convert_from = 'WORLD'
        node_new.convert_to = 'CAMERA'
        node_new.location = (-397.0209045410156, 594.7037353515625)
        node_new.name = 'Vector Transform'
        node_new.select = False
        node_new.vector_type = 'VECTOR'
        node_new.width = 140.0
        node_new.inputs[0].default_value = [0.5, 0.5, 0.5]
        node_new.outputs[0].default_value = [0.0, 0.0, 0.0]

        node_new = nodes.new(type='ShaderNodeNewGeometry')
        node_new.active_preview = False
        node_new.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        node_new.location = (-651.8067016601562, 593.0455932617188)
        node_new.name = 'Geometry'
        node_new.width = 140.0
        node_new.outputs[0].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[1].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[2].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[3].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[4].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[5].default_value = [0.0, 0.0, 0.0]
        node_new.outputs[6].default_value = 0.0
        node_new.outputs[7].default_value = 0.0
        node_new.outputs[8].default_value = 0.0

        links.new(nodes["Light Path"].outputs[0], nodes["Mix Shader"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[0], nodes["Math"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[1], nodes["Math.002"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[2], nodes["Math.001"].inputs[0])    
        links.new(nodes["Vector Transform"].outputs[0], nodes["Separate XYZ"].inputs[0])    
        links.new(nodes["Combine XYZ"].outputs[0], nodes["Emission"].inputs[0])    
        links.new(nodes["Math"].outputs[0], nodes["Combine XYZ"].inputs[0])    
        links.new(nodes["Math.002"].outputs[0], nodes["Combine XYZ"].inputs[1])    
        links.new(nodes["Math.001"].outputs[0], nodes["Combine XYZ"].inputs[2])    
        links.new(nodes["Transparent BSDF"].outputs[0], nodes["Mix Shader"].inputs[1])    
        links.new(nodes["Emission"].outputs[0], nodes["Mix Shader"].inputs[2])    
        links.new(nodes["Mix Shader"].outputs[0], nodes["Material Output"].inputs[0])    
        links.new(nodes["Geometry"].outputs[1], nodes["Vector Transform"].inputs[0])    

        self.my_material[material_name] = mat

        # 8 去除除了背景的所有object，去除所有默认材质
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.meshes.remove(obj.data)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.objects.remove(obj, do_unlink=True)
        for mat in bpy.data.materials:
            name = mat.name.split('.')
            if name[0] == 'Material':
                bpy.data.materials.remove(mat)

    def render(self,image_name="tmp",image_path=RENDERING_PATH):
        '''渲染，支持IR RGB Mask NOCS Normal '''
        if not self.model_loaded:
            print("Model not loaded.")
            return
        
        def lightModelSelect(light_mode):
            '''选择灯光模式'''
            if light_mode=="RGB":
                self.light_emitter.hide_render=True
                tmp_random_energy=random.uniform(setting["LIGHT_ENV_MAP_ENERGY_RGB"] * 0.8, setting["LIGHT_ENV_MAP_ENERGY_RGB"] * 1.2)
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = tmp_random_energy
            elif light_mode == "IR":
                self.light_emitter.hide_render = False
                # set the environment map energy
                tmp_random_energy = random.uniform(setting["LIGHT_ENV_MAP_ENERGY_IR"] * 0.8, setting["LIGHT_ENV_MAP_ENERGY_IR"] * 1.2)
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = tmp_random_energy
            elif light_mode == "Mask" or light_mode == "NOCS" or light_mode == "Normal":
                self.light_emitter.hide_render = True
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
            else:
                print("Not support the mode!")   
        def outputModelSelect(output_mode):
            '''输出模式选择'''
            if output_mode=="RGB":
                self.render_context.image_settings.file_format='PNG'
                self.render_context.image_settings.compression=0
                self.render_context.image_settings.color_mode='RGB'
                self.render_context.image_settings.color_depth='8'
                bpy.context.scene.view_settings.view_transform='Filmic'
                bpy.context.scene.render.filter_size=1.5
                self.render_context.resolution_x=640
                self.render_context.resolution_y=360
            elif output_mode=="IR":
                self.render_context.image_settings.file_format='PNG'
                self.render_context.image_settings.compression=0
                self.render_context.image_settings.color_mode='BW'
                self.render_context.image_settings.color_depth='8'
                bpy.context.scene.view_settings.view_transform='Filmic'
                bpy.context.scene.render.filter_size=1.5
                self.render_context.resolution_x=640
                self.render_context.resolution_y=360
            elif output_mode=="Mask":
                # self.render_context.image_settings.file_format='OPEN_EXR'
                self.render_context.image_settings.file_format='PNG'
                self.render_context.image_settings.color_mode='RGB'
                self.render_context.image_settings.color_depth = '8'
                bpy.context.scene.view_settings.view_transform='Raw'
                bpy.context.scene.render.filter_size=0
                self.render_context.resolution_x=640
                self.render_context.resolution_y=360
            elif output_mode=="NOCS":
                self.render_context.image_settings.file_format='PNG'
                self.render_context.image_settings.color_mode='RGB'
                self.render_context.image_settings.color_depth = '8'
                bpy.context.scene.view_settings.view_transform='Raw'
                bpy.context.scene.render.filter_size=0
                self.render_context.resolution_x=640
                self.render_context.resolution_y=360     
            elif output_mode=="Normal":
                # self.render_context.image_settings.file_format='OPEN_EXR'
                self.render_context.image_settings.file_format='PNG'
                self.render_context.image_settings.color_mode='RGB'
                self.render_context.image_settings.color_depth = '8'
                bpy.context.view_layer.use_pass_normal = True
                bpy.context.scene.view_settings.view_transform='Raw'
                bpy.context.scene.render.filter_size=1.5
                self.render_context.resolution_x=640
                self.render_context.resolution_y=360  
            else:
                print("Not support the mode!")    
        def renderEngineSelect(engine_mode):
            '''选择渲染引擎'''
            if engine_mode=="CYCLES":
                self.render_context.engine='CYCLES'
                bpy.context.scene.cycles.progressive='BRANCHED_PATH'
                bpy.context.scene.cycles.use_denoising=True
                bpy.context.scene.cycles.denoiser='NLM'
                bpy.context.scene.cycles.film_exposure=1.0
                bpy.context.scene.cycles.aa_samples=64

                bpy.context.preferences.addons["cycles"].preferences.compute_device_type="CUDA"#OPENCL
                tmp_cuda_devices,_=bpy.context.preferences.addons["cycles"].preferences.get_devices()
                #print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
                for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                    d["use"]=1#使用GPU,CPU
                    # print(d["name"],d["use"])
                tmp_device_list=setting["DEVICE_LIST"]
                tmp_activate_gpus=[]
                for i ,device in enumerate(tmp_cuda_devices):
                    if (i in tmp_device_list):
                        device.use=True
                        tmp_activate_gpus.append(device.name)
                    else:
                        device.use=False
            elif engine_mode=="EEVEE":
                self.render_context.engine='BLENDER_EEVEE'
            else:
                print("Not support the mode!")    
                
        if self.render_mode=="IR":
            bpy.context.scene.use_nodes=False
            lightModelSelect("IR")
            outputModelSelect("IR")
            renderEngineSelect("CYCLES")
        elif self.render_mode=="RGB":
            bpy.context.scene.use_nodes=False
            lightModelSelect("RGB")
            outputModelSelect("RGB")
            renderEngineSelect("CYCLES")
        elif self.render_mode=="Mask":
            bpy.context.scene.use_nodes=False
            lightModelSelect("Mask")
            outputModelSelect("Mask")
            renderEngineSelect("CYCLES")    
            bpy.context.scene.cycles.use_denoising=False
            bpy.context.scene.cycles.aa_samples=1 
        elif self.render_mode=="NOCS":
            bpy.context.scene.use_nodes=False
            lightModelSelect("NOCS")
            outputModelSelect("NOCS")
            renderEngineSelect("CYCLES")    
            bpy.context.scene.cycles.use_denoising=False
            bpy.context.scene.cycles.aa_samples=1   
        elif self.render_mode=="Normal":
            bpy.context.scene.use_nodes=True
            self.fileOutput.base_path=image_path
            self.fileOutput.file_slots[0].path=image_name[:10]+'depth_#'

            self.normal_file_output.base_path=image_path
            self.normal_file_output.file_slots[0].path=image_name[:10]+'normal_#'

            lightModelSelect("Normal")
            outputModelSelect("Normal")
            renderEngineSelect("CYCLES")    
            bpy.context.scene.cycles.use_denoising=False
            bpy.context.scene.cycles.aa_samples=32
        else:
            print("The render mode is not supported")
            return 
        bpy.context.scene.render.filepath=os.path.join(image_path,image_name)#设置渲染后的保存路径
        bpy.ops.render.render(write_still=True)

    def run(self):
        '''加载CAD模型，开始渲染'''
        # 1、加载CAD模型文件夹
        def generateCADmodeList(cad_model_path):
            tmp_CAD_model_list={}
            for tmp_class_folder in os.listdir(cad_model_path):
                if tmp_class_folder[0]=='.':
                    continue
                tmp_class_path=os.path.join(cad_model_path,tmp_class_folder)
                tmp_class_name=g_shape_synset_name_pairs[tmp_class_folder] if tmp_class_folder in g_shape_synset_name_pairs else 'other' 

                tmp_class_list=[]
                for tmp_instance_folder in os.listdir(tmp_class_path):
                    if tmp_instance_folder[0]=='.':
                        continue
                    tmp_instance_path=os.path.join(tmp_class_path,tmp_instance_folder,"model.obj")
                    tmp_class_list.append([tmp_instance_path,tmp_class_name])
                
                if tmp_class_name=='other' and 'other' in tmp_CAD_model_list:
                    tmp_CAD_model_list[tmp_class_name]=tmp_CAD_model_list[tmp_class_name]+tmp_class_list
                else:
                    tmp_CAD_model_list[tmp_class_name]=tmp_class_list
            return tmp_CAD_model_list
        CAD_model_list=generateCADmodeList(setting["CAD_model_root_path"])

        # 2 设置每个场景下的渲染帧数下的相机环境地图和背景材料
        camera_quaternion_list,camera_translation_list,envmap_id_list,envmap_rotation_elurz_list,background_material_list=[],[],[],[],[]

        def getRTfromA2B(pointCloudA,pointCloudB):
            '''计算两个点云的变换'''
            muA=np.mean(pointCloudA,axis=0)
            muB=np.mean(pointCloudB,axis=0)

            zeroMeanA=pointCloudA-muA
            zeroMeanB=pointCloudB-muB

            convMat=np.matmul(np.transpose(zeroMeanA),zeroMeanB)
            U,S,Vt=np.linalg.svd(convMat)
            R=np.matmul(Vt.T,U.T)

            if np.linalg.det(R)<0:
                print("Reflection detected")
                Vt[2,:]*=-1
                R=Vt.T*U.T
            T=(-np.matmul(R,muA.T)+muB.T).reshape(3,1)
            return R,T

        def quaternionFromRotMat(rotation_matrix):
            '''旋转矩阵转换成四元数'''
            rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
            w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
            x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
            y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
            z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
            a = [w,x,y,z]
            m = a.index(max(a))
            if m == 0:
                x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
                y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
                z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
            if m == 1:
                w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
                y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
                z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
            if m == 2:
                w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
                x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
                z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
            if m == 3:
                w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
                x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
                y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
            quaternion = (w,x,y,z)
            return quaternion

        def cameraPositionRandom(start_point_range,look_at_range,up_range):
            '''随机产生相机位置'''
            r_range,vector_range=start_point_range
            r_min,r_max=r_range
            x_min,x_max,y_min,y_max=vector_range
            r=random.uniform(r_min,r_max)
            x=random.uniform(x_min,x_max)
            y=random.uniform(y_min,y_max)
            z=math.sqrt(1-x**2-y**2)
            vector_camera_axis=np.array([x,y,z])

            x_min,x_max,y_min,y_max=up_range
            x=random.uniform(x_min,x_max)
            y=random.uniform(y_min,y_max)
            z=math.sqrt(1-x**2-y**2)
            up=np.array([x,y,z])

            x_min,x_max,y_min,y_max,z_min,z_max=look_at_range
            x=random.uniform(x_min,x_max)
            y=random.uniform(y_min,y_max)
            z=random.uniform(z_min,z_max)
            look_at=np.array([x,y,z])
            
            # 如何计算的 TODO
            position=look_at+r*vector_camera_axis
            vectorZ=-(look_at-position)/np.linalg.norm(look_at-position)
            vectorX=np.cross(up,vectorZ)/np.linalg.norm(np.cross(up,vectorZ))
            vectorY=np.cross(vectorZ,vectorX)/np.linalg.norm(np.cross(vectorZ,vectorX))

            pointSensor=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])#相机坐标系中点的位置
            pointWorld=np.array([position,position+vectorX,position+vectorY*2,position+vectorZ*3])#在世界坐标系中的位置

            resR,resT=getRTfromA2B(pointSensor,pointWorld)
            resQ=quaternionFromRotMat(resR)

            return resQ,resT

        for i in range(setting["num_frame_per_scene"]):
            camera_quaternion,camera_translation=cameraPositionRandom(setting["start_point_range"],setting["look_at_range"],setting["up_range"])
            camera_quaternion_list.append(camera_quaternion)
            camera_translation_list.append(camera_translation)

            envmap_id_list.append(random.randint(0,len(self.env_map)-1))
            envmap_rotation_elurz_list.append(random.uniform(-math.pi,math.pi))

            if setting["my_material_randomize_mode"]=='raw':
                background_material_list.append(self.my_material['default_background'])
            else:
                background_material_list.append(random.sample(self.my_material['background'],1)[0])
        
        # 3 从CAD模型文件夹中加载模型 输出meta.txt
        select_model_list_other,select_model_list_transparent,select_model_list_dis=[],[],[]
        select_number=1
        for item in CAD_model_list:
            if item in ['bottle','bowl','mug','biotube','biobottle','bioxitou']:
                for model in random.sample(CAD_model_list[item],select_number):
                    select_model_list_transparent.append(model)
            elif item in ['other']:
                for model in random.sample(CAD_model_list[item],min(3,len(CAD_model_list[item]))):
                    select_model_list_dis.append(model)
            else:
                for model in random.sample(CAD_model_list[item],select_number):
                    select_model_list_other.append(model) 
        # select_model_list_other= random.sample(select_model_list_other,random.randint(1,4))
        # select_model_list_dis= random.sample(select_model_list_dis,random.randint(1,3))
        select_model_list=select_model_list_transparent+select_model_list_other+select_model_list_dis
          
        def loadModel(file_path):
            self.model_loaded=True
            try:
                if file_path.endswith('obj'):
                    bpy.ops.import_scene.obj(filepath=file_path)
                elif file_path.endswith('3ds'):
                    bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
                elif file_path.endswith('dae'):
                    bpy.ops.wm.collada_import(filepath=file_path)
                else:
                    self.model_loaded=False
                    raise Exception("Loading failed %s"%(file_path))
            except Exception:
                self.model_loaded=False 

        def setModelPosition(instance,position_limit,instance_mask_id):
            '''随机设置示例模型的位置'''
            x_min,x_max,y_min,y_max,z=position_limit
            instance.rotation_mode='XYZ'
            instance.rotation_euler = (random.uniform(math.pi/2 - math.pi/4, math.pi/2 + math.pi/4), random.uniform(- math.pi/4, math.pi/4), random.uniform(-math.pi, math.pi))
            instance.location = (random.uniform(x_min, x_max), random.uniform(y_min, y_max), z + instance_mask_id * 0.1)
        
        def setRigidBody(instance):
            '''对实例物体增加刚体属性'''
            bpy.context.view_layer.objects.active=instance
            object_single=bpy.context.active_object
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.mass=1
            bpy.context.object.rigid_body.kinematic=True
            bpy.context.object.rigid_body.collision_shape='CONVEX_HULL'
            bpy.context.object.rigid_body.restitution=0.01
            bpy.context.object.rigid_body.angular_damping=0.8
            bpy.context.object.rigid_body.linear_damping=0.99

            bpy.context.object.rigid_body.kinematic=False
            object_single.keyframe_insert(data_path='rigid_body.kinematic',frame=0)

        def generateMaterialType(obj_name):
            '''随机选择材质'''
            flag=random.randint(0,3)
            if flag==0:
                flag=random.randint(0,1)
                if flag==0:
                    return 'raw'
                else:
                    if obj_name.split('_')[1] in setting["class_material_pairs"]["transparent"]:
                        return 'diffuse'
            else:
                flag=random.randint(0,2)
                if flag<2:
                    if obj_name.split('_')[1] in setting["class_material_pairs"]["transparent"]:
                        return 'transparent'
                    else:
                        flag=2
                if flag==2:
                    if obj_name.split('_')[1] in setting["class_material_pairs"]["specular"]:
                        return 'specular'
                    else:
                        return 'diffuse'
            return 'raw'
      
        def setMaterialRandomizeMode(class_material_pairs,material_randomize_mode,instance,material_type_in_mixed_mode):
            '''根据随机选择的材质设置材质'''
            if material_randomize_mode=='transparent' and instance.name.split('_')[1] in class_material_pairs['transparent']:
                # print(instance.name,'material mode:  transparent')
                instance.data.materials.clear()
                instance.active_material=random.sample(self.my_material['transparent'],1)[0]
            
            elif material_randomize_mode=='specular' and instance.name.split('_')[1] in class_material_pairs['specular']:
                # print(instance.name,'material mode:  specular')
                material=random.sample(self.my_material['specular'],1)[0]
                self.material.setModifyMaterial(instance,material)
            
            elif material_randomize_mode=='mixed':
                if material_type_in_mixed_mode=='diffuse' and instance.name.split('_')[1] in class_material_pairs['diffuse']:
                    # print(instance.name, 'material mode: diffuse')
                    material = random.sample(self.my_material['diffuse'], 1)[0]
                    self.material.setModifyMaterial(instance, material)
                elif material_type_in_mixed_mode == 'transparent' and instance.name.split('_')[1] in class_material_pairs['transparent']:
                    # print(instance.name, 'material mode: transparent')
                    instance.data.materials.clear()
                    instance.active_material = random.sample(self.my_material['transparent'],1)[0]
                elif material_type_in_mixed_mode == 'specular' and instance.name.split('_')[1] in class_material_pairs['specular']:
                    # print(instance.name, 'material mode: specular')
                    material = random.sample(self.my_material['specular'], 1)[0]
                    self.material.setModifyMaterial(instance, material)
                else:
                    # print(instance.name, 'material mode: raw')
                    self.material.setModifyRawMaterial(instance)
            else:
                # print(instance.name, 'material mode: raw')
                self.material.setModifyRawMaterial(instance)

        instance_id=1
        meta_output={}
        for model in select_model_list:#设置模型
            class_name=model[1]
            class_folder=model[0].split('/')[-3]
            instance_path=model[0]
            instance_folder=model[0].split('/')[-2]
            instance_name=str(instance_id)+"_"+class_name+"_"+class_folder+"_"+instance_folder
            
            # 加载模型
            loadModel(instance_path)
            obj=bpy.data.objects['model']
            obj.name=instance_name
            obj.data.name=instance_name

            # 设置模型位置
            setModelPosition(obj,(-0.3,0.3,-0.3,0.3,setting["background_position"][2]+0.2),instance_id)
            setRigidBody(obj)
            material_type_in_mixed_mode=generateMaterialType(instance_name)
            setMaterialRandomizeMode(setting["class_material_pairs"],setting["my_material_randomize_mode"],obj,material_type_in_mixed_mode)

            # 缩放模型大小
            class_scale=random.uniform(setting["g_synset_name_scale_pairs"][class_name][0],setting["g_synset_name_scale_pairs"][class_name][1])
            obj.scale=(class_scale,class_scale,class_scale)

            # 输出meta.txt
            meta_output[str(instance_id)]=[str(setting["g_synset_name_label_pairs"][class_name]),class_folder,instance_folder,str(class_scale),str(setting["material_name_label_pairs"][material_type_in_mixed_mode])]

            instance_id+=1
        
        # 4 设置好关键帧
        scene=bpy.data.scenes['Scene']
        scene.frame_start=0
        scene.frame_end=121

        render_output_file=PATH_SCENE
        for i in range(scene.frame_start,scene.frame_end+1):
            scene.frame_set(i)
            if i==120:
                break
        
        def quaternionToRotation(q):
            w, x, y, z = q
            r00 = 1 - 2 * y ** 2 - 2 * z ** 2
            r01 = 2 * x * y + 2 * w * z
            r02 = 2 * x * z - 2 * w * y

            r10 = 2 * x * y - 2 * w * z
            r11 = 1 - 2 * x ** 2 - 2 * z ** 2
            r12 = 2 * y * z + 2 * w * x

            r20 = 2 * x * z + 2 * w * y
            r21 = 2 * y * z - 2 * w * x
            r22 = 1 - 2 * x ** 2 - 2 * y ** 2
            r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
            return r

        def rotVector(q, vector_ori):
            r = quaternionToRotation(q)
            x_ori = vector_ori[0]
            y_ori = vector_ori[1]
            z_ori = vector_ori[2]
            x_rot = r[0][0] * x_ori + r[1][0] * y_ori + r[2][0] * z_ori
            y_rot = r[0][1] * x_ori + r[1][1] * y_ori + r[2][1] * z_ori
            z_rot = r[0][2] * x_ori + r[1][2] * y_ori + r[2][2] * z_ori
            return (x_rot, y_rot, z_rot)

        def cameraLPosToCameraRPos(q_l,pos_l,baseline_dis):
            '''左相机位置到右相机位置'''
            vector_camera_l_y=(1,0,0)
            vector_rot = rotVector(q_l, vector_camera_l_y)
            pos_r = (pos_l[0] + vector_rot[0] * baseline_dis,
                    pos_l[1] + vector_rot[1] * baseline_dis,
                    pos_l[2] + vector_rot[2] * baseline_dis)
            return pos_r
        
        def setCamera(quaternion,translation,fov,baseline_distance):
            '''设置相机位置'''
            self.camera_l.data.angle=fov
            self.camera_r.data.angle=self.camera_l.data.angle

            cx,cy,cz=translation[0],translation[1],translation[2]
            self.camera_l.location[0]=cx
            self.camera_l.location[1]=cy
            self.camera_l.location[2]=cz
            self.camera_l.rotation_mode='QUATERNION'
            self.camera_l.rotation_quaternion[0]=quaternion[0]
            self.camera_l.rotation_quaternion[1]=quaternion[1]
            self.camera_l.rotation_quaternion[2]=quaternion[2]
            self.camera_l.rotation_quaternion[3]=quaternion[3]

            cx,cy,cz=cameraLPosToCameraRPos(quaternion,(cx,cy,cz),baseline_distance)
            self.camera_r.location[0]=cx
            self.camera_r.location[1]=cy
            self.camera_r.location[2]=cz
            self.camera_r.rotation_mode='QUATERNION'
            self.camera_r.rotation_quaternion[0]=quaternion[0]
            self.camera_r.rotation_quaternion[1]=quaternion[1]
            self.camera_r.rotation_quaternion[2]=quaternion[2]
            self.camera_r.rotation_quaternion[3]=quaternion[3]
        
        def checkVisible(threshold=(0.1,0.9,0.1,0.9)):
            '''检查可视化'''
            w_min,w_max,h_min,h_max=threshold
            tmp_visible_objects_list=[]
            bpy.context.view_layer.update()
            cs,ce=self.camera_l.data.clip_start,self.camera_l.data.clip_end
            for obj in bpy.data.objects:
                if obj.type=='MESH' and not obj.name.split('_')[0]=='background':
                    obj_center=obj.matrix_world.translation
                    co_ndc=world_to_camera_view(scene,self.camera_l,obj_center)
                    if w_min<co_ndc.x<w_max and h_min<co_ndc.y<h_max and cs<co_ndc.z<ce:
                        obj.select_set(True)
                        tmp_visible_objects_list.append(obj)
                else:
                    obj.select_set(False)
            return tmp_visible_objects_list
        
        def quaternionMul(q1, q2):
            s1 = q1[0]
            v1 = np.array(q1[1:])
            s2 = q2[0]
            v2 = np.array(q2[1:])
            s = s1 * s2 - np.dot(v1, v2)
            v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
            return (s, v[0], v[1], v[2])

        def getInstancePose():
            '''获取实例位置'''
            tmp_instance_pose={}
            bpy.context.view_layer.update()
            cam=self.camera_l
            for obj in bpy.data.objects:
                if obj.type=='MESH' and not obj.name.split('_')[0]=='background':
                    tmp_instance_id=obj.name.split('_')[0]
                    tmp_mat_rel=cam.matrix_world.inverted() @ obj.matrix_world

                    tmp_relative_location=[tmp_mat_rel.translation[0],-tmp_mat_rel.translation[1],-tmp_mat_rel.translation[2]]
                    tmp_relative_rotation_quat=[tmp_mat_rel.to_quaternion()[0],tmp_mat_rel.to_quaternion()[1],tmp_mat_rel.to_quaternion()[2],tmp_mat_rel.to_quaternion()[3]]

                    tmp_quat_x=[0,1,0,0]
                    tmp_quat=quaternionMul(tmp_quat_x,tmp_relative_rotation_quat)
                    tmp_quat=[tmp_quat[0], -tmp_quat[1],-tmp_quat[2],-tmp_quat[3]]

                    tmp_instance_pose[str(tmp_instance_id)]=[tmp_quat,tmp_relative_location]
            return tmp_instance_pose

        # 5 生成meta.txt
        visible_objects_list=[]
        instance_pose_list=[]
        visible_threshold=(0.03,0.97,0.05,0.95)
        for i in range(setting["num_frame_per_scene"]):
            setCamera(camera_quaternion_list[i],camera_translation_list[i],setting["camera_fov"],setting["baseline_distance"])
            visible_objects_list.append(checkVisible(visible_threshold))
            instance_pose_list.append(getInstancePose())
        
        for i in range(setting["num_frame_per_scene"]):
            path_meta=os.path.join(PATH_SCENE,str(i).zfill(9)+"_meta.txt")
            if os.path.exists(path_meta):
                os.remove(path_meta)
            file_write_obj=open(path_meta,'w')
            for index in meta_output:
                file_write_obj.write(index)
                file_write_obj.write(' ')
                for item in meta_output[index]:
                    file_write_obj.write(item)
                    file_write_obj.write(' ')
                for item in instance_pose_list[i][index]:
                    for var in item:
                        file_write_obj.write(str(var))
                        file_write_obj.write(' ')
                file_write_obj.write('\n')
            file_write_obj.close()

        def setVisiableObjects(visiable_objects_list):
            '''设置物体是否可见'''
            for obj in bpy.data.objects:
                if obj.type=='MESH' and not obj.name.split('_')[0]=='background':
                    if obj in visiable_objects_list:
                        obj.hide_render=False
                    else:
                        obj.hide_render=True

        def setLighting():
            '''设置灯光'''
            self.light_emitter.location=self.camera_l.location+0.51*(self.camera_r.location-self.camera_l.location)
            self.light_emitter.rotation_mode='QUATERNION'
            self.light_emitter.rotation_quaternion=self.camera_r.rotation_quaternion

            bpy.context.view_layer.objects.active=None
            self.render_context.engine='CYCLES'
            self.light_emitter.select_set(True)
            self.light_emitter.data.use_nodes=True
            self.light_emitter.data.type='POINT'
            self.light_emitter.data.shadow_soft_size=0.001
            random_energy=random.uniform(setting["LIGHT_EMITTER_ENERGY"]*0.9,setting["LIGHT_EMITTER_ENERGY"]*1.1)
            self.light_emitter.data.energy=random_energy

            # 去除默认节点增加新节点
            light_emitter=bpy.data.objects['light_emitter'].data
            light_emitter.node_tree.nodes.clear()

            light_output = light_emitter.node_tree.nodes.new("ShaderNodeOutputLight")
            node_1 = light_emitter.node_tree.nodes.new("ShaderNodeEmission")
            node_2 = light_emitter.node_tree.nodes.new("ShaderNodeTexImage")
            node_3 = light_emitter.node_tree.nodes.new("ShaderNodeMapping")
            node_4 = light_emitter.node_tree.nodes.new("ShaderNodeVectorMath")
            node_5 = light_emitter.node_tree.nodes.new("ShaderNodeSeparateXYZ")
            node_6 = light_emitter.node_tree.nodes.new("ShaderNodeTexCoord")

            light_emitter.node_tree.links.new(light_output.inputs[0], node_1.outputs[0])
            light_emitter.node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
            light_emitter.node_tree.links.new(node_2.inputs[0], node_3.outputs[0])
            light_emitter.node_tree.links.new(node_3.inputs[0], node_4.outputs[0])
            light_emitter.node_tree.links.new(node_4.inputs[0], node_6.outputs[1])
            light_emitter.node_tree.links.new(node_4.inputs[1], node_5.outputs[2])
            light_emitter.node_tree.links.new(node_5.inputs[0], node_6.outputs[1])

            # 设置节点参数
            node_1.inputs[1].default_value = 1.0        # scale
            node_2.extension = 'CLIP'

            node_3.inputs[1].default_value[0] = 0.5
            node_3.inputs[1].default_value[1] = 0.5
            node_3.inputs[1].default_value[2] = 0
            node_3.inputs[2].default_value[0] = 0
            node_3.inputs[2].default_value[1] = 0
            node_3.inputs[2].default_value[2] = 0.05
            
            node_3.inputs[3].default_value[0] = 0.6
            node_3.inputs[3].default_value[1] = 0.85
            node_3.inputs[3].default_value[2] = 0
            node_4.operation = 'DIVIDE'

            node_2.image = self.pattern
        
        def setEnvMap(envmap_id,envmap_elurz):
            '''设置当前场景的环境节点'''
            node_tree=bpy.context.scene.world.node_tree
            node_environment=node_tree.nodes['Environment Texture']
            node_environment.image=self.env_map[envmap_id]
            node_mapping=node_tree.nodes['Mapping']
            node_mapping.inputs[2].default_value[2]=envmap_elurz

        # 6 渲染左右相机图像和 rgb图像
        if setting['render_mode_list']['IR'] or setting['render_mode_list']['RGB']:
            for i in range(setting['num_frame_per_scene']):
                setVisiableObjects(visible_objects_list[i])
                setCamera(camera_quaternion_list[i],camera_translation_list[i],setting['camera_fov'],setting['baseline_distance'])
                setLighting()
                setEnvMap(envmap_id_list[i],envmap_rotation_elurz_list[i])
                for obj in bpy.data.objects:
                    if obj.type=='MESH' and obj.name.split('_')[0]=='background':
                        obj.active_material=background_material_list[i]
                # 渲染左右相机图像
                if setting['render_mode_list']['IR']:
                    self.render_mode="IR"
                    scene.camera=bpy.data.objects['camera_l']
                    self.render(str(i).zfill(9)+'_ir_l',render_output_file)

                    scene.camera=bpy.data.objects['camera_r']
                    self.render(str(i).zfill(9)+'_ir_r',render_output_file)
                # 渲染RGB图像
                if setting['render_mode_list']['RGB']:
                    self.render_mode="RGB"
                    scene.camera=bpy.data.objects['camera_l']
                    self.render(str(i).zfill(9)+'_color',render_output_file)
        
        # 7 渲染掩码和深度图像
        if setting['render_mode_list']['Mask']:
            # 设置实例掩码作为材质
            for obj in bpy.data.objects:
                if obj.type=='MESH':
                    obj.data.materials.clear()
                    material_name="mask_"+obj.name.split('_')[0]
                    obj.active_material=self.my_material[material_name]
            # 渲染掩码和深度图像
            for i in range(setting['num_frame_per_scene']):
                setVisiableObjects(visible_objects_list[i])
                setCamera(camera_quaternion_list[i],camera_translation_list[i],setting['camera_fov'],setting['baseline_distance'])
                self.render_mode="Mask"
                scene.camera=bpy.data.objects['camera_l']
                self.render(str(i).zfill(9)+'_mask',render_output_file)    

        # 8 渲染法线图像            
        if setting['render_mode_list']['Normal']:  
            # 设置实例法线作为材质
            for obj in bpy.data.objects:
                if obj.type=='MESH':
                    obj.data.materials.clear()
                    obj.active_material=self.my_material["normal"]

            # 渲染法线图像
            for i in range(setting['num_frame_per_scene']):
                setVisiableObjects(visible_objects_list[i])
                setCamera(camera_quaternion_list[i],camera_translation_list[i],setting['camera_fov'],setting['baseline_distance'])
                self.render_mode="Normal"
                scene.camera=bpy.data.objects['camera_l']
                self.render(str(i).zfill(9)+'_normal',render_output_file)   
          
        # 9 渲染Nocs图像
        if setting['render_mode_list']['NOCS']:
            # 重新设置背景顶点颜色
            for obj in bpy.data.objects:
                if obj.type=='MESH' and obj.name.split('_')[0]=='background':
                    start_time_obj=time.time()
                    vertex_colors=obj.data.vertex_colors
                    while vertex_colors:
                        vertex_colors.remove(vertex_colors[0])
                    obj.data.update()
                    obj.data.vertex_colors.new(name='Col_R',do_init=False)
                    obj.data.vertex_colors.new(name='Col_G',do_init=False)
                    obj.data.vertex_colors.new(name='Col_B',do_init=False)
                    vcol_layer_r=obj.data.vertex_colors['Col_R']
                    vcol_layer_g=obj.data.vertex_colors['Col_G']
                    vcol_layer_b=obj.data.vertex_colors['Col_B']

                    count=0
                    start_time_loop=time.time()
                    for loop_index,loop in enumerate(obj.data.loops):
                        vcol_layer_r.data[loop_index].color=Vector([0,0,0,1])
                        vcol_layer_g.data[loop_index].color=Vector([0,0,0,1])
                        vcol_layer_b.data[loop_index].color=Vector([0,0,0,1])
                        count+=1
                    end_time_obj=time.time()
                    obj.data.vertex_colors.active=vcol_layer_r
                    obj.data.update()
                    #print(obj.name, ' time: ', end_time_obj - start_time_obj, 'mean time: ', (end_time_obj - start_time_loop)/count)
                # 重新设置物体顶点颜色
                if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                    start_time_obj = time.time()
                    vertex_colors = obj.data.vertex_colors
                    # remove exists vertex colors
                    while vertex_colors:
                        vertex_colors.remove(vertex_colors[0])
                    obj.data.update()

                    # create new vertex color layer
                    obj.data.vertex_colors.new(name='Col_R', do_init=True)
                    obj.data.vertex_colors.new(name='Col_G', do_init=True)
                    obj.data.vertex_colors.new(name='Col_B', do_init=True)
                    vcol_layer_r = obj.data.vertex_colors['Col_R']
                    vcol_layer_g = obj.data.vertex_colors['Col_G']
                    vcol_layer_b = obj.data.vertex_colors['Col_B']


                    count = 0
                    start_time_loop = time.time()
                    for loop_index, loop in enumerate(obj.data.loops):
                        loop_vert_index = loop.vertex_index
                        # here the scale is manually set for the cube to normalize it within [-0.5, 0.5]
                        scale = 1
                        color_x = scale * obj.data.vertices[loop_vert_index].co.x + 0.5
                        color_y = scale * obj.data.vertices[loop_vert_index].co.y + 0.5
                        color_z = scale * obj.data.vertices[loop_vert_index].co.z + 0.5
                        vcol_layer_r.data[loop_index].color = Vector([0, 0, 0, color_x])
                        vcol_layer_g.data[loop_index].color = Vector([0, 0, 0, color_y])
                        vcol_layer_b.data[loop_index].color = Vector([0, 0, 0, 1 - color_z])
                        count += 1
                    end_time_obj = time.time()
                    obj.data.vertex_colors.active = vcol_layer_r
                    obj.data.update()
                    #print(obj.name, ' time: ', end_time_obj - start_time_obj, 'mean time: ', (end_time_obj - start_time_loop)/count)
            
            # 设置实例nocs作为材质
            for obj in bpy.data.objects:
                if obj.type=='MESH':
                    obj.data.materials.clear()
                    obj.active_material=self.my_material["coord_color"]
            # 渲染nocs图像
            for i in range(setting['num_frame_per_scene']):
                setVisiableObjects(visible_objects_list[i])
                setCamera(camera_quaternion_list[i],camera_translation_list[i],setting['camera_fov'],setting['baseline_distance'])
                self.render_mode="NOCS"
                scene.camera=bpy.data.objects['camera_l']
                self.render(str(i).zfill(9)+'_coord',render_output_file)   
        
        # 10 动态清除上下文    
        contex=bpy.context
        for ob in contex.selectable_objects:
            ob.animation_data_clear()
        #print(bpy.data.materials)
        #print(len(bpy.data.materials))
        
if __name__=='__main__':
    render=BlenderRender(viewport_size_x=setting['camera_width'], viewport_size_y=setting['camera_height'])
    render.run()