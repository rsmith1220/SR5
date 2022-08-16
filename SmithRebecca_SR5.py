#Rebecca Smith
#Seccion 20

from gl import Renderer, color, V2, V3
from textures import Texture
from obj import Obj
import random

from shaders import flat,unlit, gourad, toon, glow, textureBlend


w=900
h=900
z=-10

rend= Renderer(w,h)

rend.dirLight = V3(1,0,0)

rend.active_texture = Texture("body.bmp")
rend.active_texture2 = Texture("shades.bmp")
rend.active_shader = textureBlend


rend.glLoadModel("earth.obj",
                translate = V3(w/2, h/2, z/2),
                rotate = V3(0, 180, 0),
                scale = V3(300,300,300))

rend.glFinish("output.bmp")

