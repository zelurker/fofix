from __future__ import division
#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2009 Vlad Emelyanov                                 #
#                                                                   #
# This program is free software; you can redistribute it and/or     #
# modify it under the terms of the GNU General Public License       #
# as published by the Free Software Foundation; either version 2    #
# of the License, or (at your option) any later version.            #
#                                                                   #
# This program is distributed in the hope that it will be useful,   #
# but WITHOUT ANY WARRANTY; without even the implied warranty of    #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     #
# GNU General Public License for more details.                      #
#                                                                   #
# You should have received a copy of the GNU General Public License #
# along with this program; if not, write to the Free Software       #
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,        #
# MA  02110-1301, USA.                                              #
#####################################################################

from builtins import range
from future.utils import raise_
from builtins import object
from past.utils import old_div
from OpenGL.GL import *

import os
import sys
import string
from random import random
import time
import Log
import pygame.image
import Config
import Version

#OGL constants for compatibility with all PyOpenGL versions
#now multitexturing should work even in PyOpenGL 2.x, if your card supports ARB ext
#stump: managed to eliminate the need for every one of these.  If I was wrong, please uncomment only the necessary line and notify me.
#GL_TEXTURE_3D = 32879
#GL_TEXTURE_WRAP_R = 32882
#GL_TEXTURE0_ARB, GL_TEXTURE1_ARB, GL_TEXTURE2_ARB, GL_TEXTURE3_ARB = 33984, 33985, 33986, 33987
#GL_FRAGMENT_SHADER_ARB = 0x8B30
#GL_VERTEX_SHADER_ARB = 0x8B31
#GL_OBJECT_COMPILE_STATUS_ARB= 0x8B81
#GL_OBJECT_LINK_STATUS_ARB = 0x8B82
#GL_INFO_LOG_LENGTH_ARB = 0x8B84
#GL_CLAMP_TO_EDGE = 33071

# stump: these don't throw the exception for being unsupported - that happens later.
# There has to be an active OpenGL context already for the checking to occur.
# We do the check down in ShaderList.set().
# evilynux : Wrapped around try/except as pyopengl < 2.0.2.x fails here.
try:
  from OpenGL.GL.ARB.shader_objects import *
  from OpenGL.GL.ARB.vertex_shader import *
  from OpenGL.GL.ARB.fragment_shader import *
  from OpenGL.GL.ARB.multitexture import *
  from OpenGL.GL.EXT.texture3D import *
except:
  Log.warn("Importing OpenGL ARB fails therefore shaders are disabled."\
             " It is most likely that your version of PyOpenGL is too old.")
  glInitShaderObjectsARB = lambda: False

class ShaderCompilationError(Exception):
  pass


#stump: apply some fixups for pyOpenGL 2.x if necessary.
def checkFunctionsForPyOpenGL2x():
  global glShaderSourceARB
  global glGetObjectParameterivARB

  # Check the version of pyOpenGL.
  import OpenGL
  if OpenGL.__version__ < '3':
    #stump: the binding of glShaderSourceARB() in pyOpenGL 2.x segfaults on use... ugh!
    # (It and glGetObjectParameterivARB() also have incompatible declarations - let's fix that too.)

    try:
      from ctypes import c_int, c_char_p, POINTER, cdll, byref
      if os.name == 'nt':
        from ctypes import WINFUNCTYPE, windll
      elif sys.platform == 'darwin':
        from ctypes.util import find_library
        
    except ImportError:
      raise ShaderCompilationError('ctypes is required to use shaders with pyOpenGL 2.x.')
    else:

      if os.name == 'nt':  # Windows - look for the functions using wglGetProcAddress()

        # Grab the function pointers the standard Windows way.
        # (opengl32.dll doesn't directly export extension entry points like other platforms' OpenGL libraries do.)
        ptr_glShaderSourceARB = windll.opengl32.wglGetProcAddress('glShaderSourceARB')
        ptr_glGetObjectParameterivARB = windll.opengl32.wglGetProcAddress('glGetObjectParameterivARB')

        # If we got good function pointers, make them callable using ctypes.
        if ptr_glShaderSourceARB:
          func_glShaderSourceARB = WINFUNCTYPE(None, c_int, c_int, POINTER(c_char_p), POINTER(c_int))(ptr_glShaderSourceARB)
        else:
          raise ShaderCompilationError('wglGetProcAddress("glShaderSourceARB") returned NULL - are shaders supported?')

        if ptr_glGetObjectParameterivARB:
          func_glGetObjectParameterivARB = WINFUNCTYPE(None, c_int, c_int, POINTER(c_int))(ptr_glGetObjectParameterivARB)
        else:
          raise ShaderCompilationError('wglGetProcAddress("glGetObjectParameterivARB") returned NULL - are shaders supported?')

      else:  # something else - assume that the OpenGL library exports the extension entry point

        # Figure out where the OpenGL library is.
        if sys.platform == 'darwin':  # Mac OS X
          glLibrary = cdll.LoadLibrary(find_library('OpenGL'))
        else: # something else; most likely GNU/Linux
          glLibrary = cdll.LoadLibrary('libGL.so')

        try:
          func_glShaderSourceARB = glLibrary.glShaderSourceARB
          func_glGetObjectParameterivARB = glLibrary.glGetObjectParameterivARB
        except:
          raise ShaderCompilationError('Cannot find glShaderSourceARB() and/or glGetObjectParameterivARB() in the OpenGL library - are shaders supported?')

      # Wrap supporting glShaderSource(shader object, iterable object) returning None, as used below.
      def glShaderSourceARB(shader, source):
        srcList = list(source)
        srcListType = c_char_p * len(srcList)
        func_glShaderSourceARB(shader, len(srcList), srcListType(*srcList), None)

      # Wrap supporting glGetObjectParameterivARB(shader object, parameter id) returning int, as used below.
      def glGetObjectParameterivARB(shader, param):
        retval = c_int(0)
        func_glGetObjectParameterivARB(shader, param, byref(retval))
        return retval.value
   
  #should be placed somewhere else  
  if sys.platform == 'darwin':
    global GL_TEXTURE_3D_EXT
    global GL_TEXTURE_WRAP_R_EXT
    global glTexImage3DEXT
    GL_TEXTURE_3D_EXT = GL_TEXTURE_3D
    GL_TEXTURE_WRAP_R_EXT = GL_TEXTURE_WRAP_R
    glTexImage3DEXT = glTexImage3D


# main class for shaders library
class ShaderList(object):
  def __init__(self):
    self.shaders = {}		# list of all shaders
    self.active = 0		# active shader
    self.texcount = 0
    self.workdir = ""		# dir that contains shader files
    self.enabled = False	# true if shaders are compiled
    self.turnon = False		# true if shaders are enabled in settings
    self.var = {}		# different variables
    self.assigned = {}		# list for shader replacement
    self.globals = {}		# list of global vars for every shader


  def make(self, fname, name = ""):
    """Compile a shader.
       fname = base filename for shader files
       name  = name to use for this shader (defaults to fname)

       Returns nothing, or raises an exception on error."""

    if name == "":
      name = fname
    fullname = os.path.join(self.workdir, fname)
    vertname, fragname = fullname+".vert", fullname+".frag"
    Log.debug('Compiling shader "%s" from %s and %s.' % (name, vertname, fragname))
    program = self.compile(open(vertname), open(fragname))
    sArray = {"program": program, "name": name, "textures": []}
    self.getVars(vertname, program, sArray)
    self.getVars(fragname, program, sArray)
    self.shaders[name] = sArray
    if "Noise3D" in self.shaders[name]:
      self.setTexture("Noise3D",self.noise3D,name)


  def compileShader(self, source, shaderType):
    """Compile shader source of given type.
       source     = file object open to shader source code
                    (or something else returning lines of GLSL code when iterated over)
       shaderType = GL_VERTEX_SHADER_ARB or GL_FRAGMENT_SHADER_ARB

       Returns the shader object, or raises an exception on error."""

    shader = glCreateShaderObjectARB(shaderType)
    glShaderSourceARB( shader, source )
    glCompileShaderARB( shader )
    status = glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB)
    if not status:
      raise_(ShaderCompilationError, self.log(shader))
    else:
      return shader


  def compile(self, vertexSource, fragmentSource):
    """Create an OpenGL program object from file objects open to the source files.
       vertexSource   = file object open to vertex shader source code
       fragmentSource = file object open to fragment shader source code

       Returns the program object, or raises an exception on error."""

    program = glCreateProgramObjectARB()
  
    vertexShader = self.compileShader(vertexSource, GL_VERTEX_SHADER_ARB)
    fragmentShader = self.compileShader(fragmentSource, GL_FRAGMENT_SHADER_ARB)

    glAttachObjectARB(program, vertexShader)  
    glAttachObjectARB(program, fragmentShader)
    glValidateProgramARB( program )
    glLinkProgramARB(program)
    glDeleteObjectARB(vertexShader)
    glDeleteObjectARB(fragmentShader)
    return program


  def getVars(self, fname, program, sArray):
    """Read the names of uniform variables from a shader file.
       fname   = shader filename
       program = OpenGL program object compiled from that shader file
       sArray  = dictionary to add variable information to"""

    for line in open(fname):
      aline = line[:string.find(line,";")]
      aline = aline.split(' ')
      if '(' in aline[0]:
        break
      if aline[0] == "uniform":
        value = None
        try:    n = int(aline[1][-1])
        except: n = 4
        if   aline[1] == "bool": value = False
        elif aline[1] == "int": value = 0
        elif aline[1] == "float": value = 0.0 
        elif aline[1][:-1] == "bvec": value = (False,)*n
        elif aline[1][:-1] == "ivec": value = (0,)*n
        elif aline[1][:-1] == "vec": value = (.0,)*n
        elif aline[1][:-1] == "mat": value = ((.0,)*n,)*n
        elif aline[1][:-2] == "sampler": 
          value, self.texcount = self.texcount, self.texcount + 1 
          if aline[1] == "sampler1D":   textype = GL_TEXTURE_1D
          elif aline[1] == "sampler2D": textype = GL_TEXTURE_2D
          elif aline[1] == "sampler3D": textype = GL_TEXTURE_3D_EXT
          sArray["textures"].append((aline[2],textype,0))
        aline[2] = aline[2].split(',')
        for var in aline[2]:
          sArray[var] = [glGetUniformLocationARB(program, var), value]


  #simplified texture binding function
  def setTexture(self,name,texture,program = None):
    if program in self.assigned:
      program = self.assigned[program]
    if program == None:  program = self.active
    else: program = self[program]
    
    for i in range(len(program["textures"])):
      if program["textures"][i][0] == name:
        program["textures"][i] = (program["textures"][i][0], program["textures"][i][1], texture)
        return True
    return False


  def getVar(self, var = "program", program = None):
    """Get a uniform variable value from a shader.
       var     = variable name
       program = shader name, or None (default) for the currently active shader

       Returns the value.  If the variable does not exist, KeyError is raised."""

    if program in self.assigned:
      program = self.assigned[program]
      
    if program is None:
      program = self.active
    else:
      program = self[program]

    if program is None or var not in program:
      return False
    else:
      return program[var][1]
      return True


  def setVar(self, var, value, program = None):
    """Set a uniform variable value for a shader.
       var     = variable name
       value   = new value
       program = shader name, or None (default) for the currently active shader

       Returns nothing.  If the variable does not exist, KeyError is raised."""

    if program in self.assigned:
      program = self.assigned[program]
    if program is None:
      program = self.active
    else:
      program = self[program]



    if program is None or var not in program:
      return
      
    if type(value) == str:
      if value in self.var:
        value = self.var[value]
      else:
        return

    pos = program[var]
    pos[1] = value
    if program == self.active:
      if type(value) == list:
        value = tuple(value)
      if type(value) == bool:
        if pos[1]: glUniform1iARB(pos[0],1)
        else: glUniform1iARB(pos[0],0)
      elif type(value) == float:
        glUniform1fARB(pos[0],pos[1])
      elif type(value) == int:
        glUniform1iARB(pos[0],pos[1])
      elif type(value) == tuple:
        if type(value[0]) == float:
          if   len(value) == 2: glUniform2fARB(pos[0],*pos[1])
          elif len(value) == 3: glUniform3fARB(pos[0],*pos[1])
          elif len(value) == 4: glUniform4fARB(pos[0],*pos[1])
        elif type(value[0]) == int:
          if   len(value) == 2: glUniform2iARB(pos[0],*pos[1])
          elif len(value) == 3: glUniform3iARB(pos[0],*pos[1])
          elif len(value) == 4: glUniform4iARB(pos[0],*pos[1])
      elif type(value) == int:
        glUniform1iARB(pos[0],pos[1])
      else:
        raise TypeError('Unsupported value type (must be bool, float, int, long, or tuple or list of float or int).')


  # slightly changes uniform variable  
  def modVar(self, var, value, effect = 0.05, alphaAmp=1.0, program = None):  
    old = self.getVar(var,program)
    if old is None:
      return False
    if type(old) == tuple:
      new = ()
      for i in range(len(old)):
        if i==3: new += (old[i] * (1-effect) + value[i] * effect*alphaAmp,)
        else: new += (old[i] * (1-effect) + value[i] * effect,)
    else:
      new = old * (1-effect) + value * effect
    self.setVar(var,new,program)
    return True
   
  # enables shader program     
  def enable(self, shader):
    if not self.turnon:
      return False

    if shader in self.assigned:
      shader = self.assigned[shader]

    if self[shader] is None:
      return False

    glUseProgramObjectARB(self[shader]["program"])
    self.active = self.shaders[shader]
    self.setTextures()
    self.update()
    self.globals["time"] = self.time()
    self.setGlobals()
    if self.getVar("time"):
      self.setVar("dt",self.globals["time"]-self.getVar("time"))
      
    return True
     
  # transmit global vars to uniforms 
  def setGlobals(self):
    for i in list(self.globals.keys()):
      self.setVar(i,self.globals[i])

  # update all uniforms        
  def update(self):
    for i in list(self.active.keys()):
      if i != "textures":
        if type(self.active[i]) == list and self.active[i][1] is not None:
          self.setVar(i,self.active[i][1])

  # set standart OpenGL program active 
  def disable(self):
    if self.active != 0:
      glUseProgramObjectARB(0)
      self.active = 0

  # return active program control
  def activeProgram(self):
    if self.active != 0:
      return self.active["name"]
    else:
      return 0


  def log(self, shader):
    """Get the error log for a shader.
       shader = object to get log from

       Returns a string containing the log or None if there is no log."""

    length = glGetObjectParameterivARB(shader, GL_INFO_LOG_LENGTH)
    if length > 0:
      log = glGetInfoLogARB(shader)
      return log


  # update and bind all textures
  def setTextures(self, program = None):
    if program in self.assigned:
      program = self.assigned[program]
    if program is None:
      program = self.active

    for i in range(len(program["textures"])):
      glActiveTextureARB(self.multiTex[i])
      glBindTexture(program["textures"][i][1], program["textures"][i][2]) 
      

      
  def makeNoise3D(self,size=32, c = 1, type = GL_RED):
    texels=[]
    for i in range(size):
      arr2 = []
      for j in range(size):
        arr = []
        for k in range(size):
          arr.append(random())
        arr2.append(arr)
      texels.append(arr2)
          
    self.smoothNoise3D(size, 2, texels)
    
    for i in range(size):
      for j in range(size):
        for k in range(size):
          texels[i][j][k] = int(255 * texels[i][j][k])

    texture = 0

    glBindTexture(GL_TEXTURE_3D_EXT, texture)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, c,size, size, size, 0, type, GL_UNSIGNED_BYTE, texels)
    return texture
    
  def makeNoise2D(self,size=64, c = 1, type = GL_RED):
    texels=[]
    for i in range(size):
      texels.append([])
      for j in range(size):
        texels[i].append(random())
    
    self.smoothNoise(size, 2, texels)
    self.smoothNoise(size, 3, texels)
    self.smoothNoise(size, 4, texels)  
    
    for i in range(size):
      for j in range(size):
        texels[i][j] = int(255 * texels[i][j])
        
    texture = 0
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, c,size, size, 0, type, GL_UNSIGNED_BYTE, texels)
    return texture
    
  def loadTex3D(self, fname, type = GL_RED):
    file = os.path.join(self.workdir,fname)
    if os.path.exists(file):
      noise = open(file).read()
      size = int(len(noise)**(1/3.0))
    else:
      Log.debug("Can't load %s; generating random 3D noise instead." % file)
      return self.makeNoise3D(16)
          
    #self.smoothNoise3D(size, 2, texels)
    #self.smoothNoise3D(size, 4, texels)
    

    texture = 0

    glBindTexture(GL_TEXTURE_3D_EXT, texture)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, 1,size, size, size, 0, type, GL_UNSIGNED_BYTE, noise)
    return texture

  def loadTex2D(self, fname, type = GL_RGB):
    file = os.path.join(self.workdir,fname)
    if os.path.exists(file):
      img = pygame.image.load(file)
      noise = pygame.image.tostring(img, "RGB")
    else:
      Log.debug("Can't load %s; generating random 2D noise instead." % fname)
      return self.makeNoise2D(16)

    texture = 0
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, 1, img.get_width(), img.get_height(), 0, type, GL_UNSIGNED_BYTE, noise)
    return texture


  def smoothNoise(self, size, c, noise):
    for x in range(size):
      for y in range(size):
        col1 = noise[x][y]
        col2 = noise[old_div(old_div(size,2),(1-c))+old_div(x,c)][old_div(old_div(size,2),(1-c))+old_div(y,c)]
        noise[x][y] = (1-1/float(c))*col1+1/float(c)*col2
    
  def smoothNoise3D(self, size, c, noise):
    for i in range(size):
      for j in range(size):
        for k in range(size):
          col1 = noise[i][j][k]
          col2 = noise[old_div(old_div(size,2),(1-c))+old_div(i,c)][old_div(old_div(size,2),(1-c))+old_div(j,c)][old_div(old_div(size,2),(1-c))+old_div(k,c)]
          noise[i][j][k] = (1-1/float(c))*col1+1/float(c)*col2
      
  def __getitem__(self, name):
    if name in self.shaders:
      return self.shaders[name]
    else:
      return None
      
  def time(self):
    return time.clock()

  def reset(self):
    self.checkIfEnabled()
    if self.turnon:
      self.var["color"] = {}                   #color for guitar neck flashing
      self.var["solocolor"] = (0.0,)*4         #color for GH3 solo lightnings
      self.var["eqcolor"] = (0.0,)*4           #color for equalizer
      self.var["fret"] = {}                    #last note hit time for each player
      self.var["fretpos"] = {}                 #last note hit pos for each player
      self.var["scoreMult"] = {}               #score multiplier for each player
      self.var["multChangePos"] = {}           #score multiplier last changing pos for each player
      
      self.globals["bpm"] = 120.0
      self.globals["breActive"] = False
      self.globals["dfActive"] = False
      self.globals["isDrum"] = False
      self.globals["isFailing"] = False
      self.globals["isMultChanged"] = False
      self.globals["killswitch"] = False
      self.globals["killswitchPos"] = -10.0
      self.globals["multChangePos"] = -10.0
      self.globals["notepos"] = -10.0
      self.globals["rockLevel"] = 0.5
      self.globals["scoreMult"] = 1
      self.globals["soloActive"] = False
      self.globals["songpos"] = 0.0
      #self.loadFromIni()
    
  # check Settings to enable, disable or assign shaders
  def checkIfEnabled(self):
    if Config.get("video","shader_use"):
      if self.enabled:
        self.turnon = True
      else:
        self.set(os.path.join(Version.dataPath(), "shaders"))
    else:
      self.turnon = False


    if self.turnon:
      for i in list(self.shaders.keys()):
        value = Config.get("video","shader_"+i)
        if value != "None":
          if value == "theme":
            if Config.get("theme","shader_"+i) == "True":
              value = i
            else:
              continue
          self.assigned[i] = value
      return True
    return False
         
  def defineConfig(self):
    for name in list(self.shaders.keys()):
      for key in list(self[name].keys()):
        Config.define("shader", name+"_"+key,  str, "None")
         
  def loadFromIni(self):
    for name in list(self.shaders.keys()):
      for key in list(self[name].keys()):
        value = Config.get("theme",name+"_"+key)
        if value != "None":
          if value == "True": value = True
          elif value == "False": value = False
          else:
            value = value.split(",")
            for i in range(len(value)):
              value[i] = float(value[i])
            if len(value) == 1: value = value[0]
            else: value = tuple(value)
          if key == "enabled":
            if Config.get("video","shader_"+name) == 2:
              self[name][key] = value
          else:
            if len(self[name][key]) == 2:
              self[name][key][1] = value


  def set(self, dir):
    """Do all shader setup.
       dir = directory to load shaders from
    """

    #stump: check whether all needed extensions are actually supported
    if not glInitShaderObjectsARB():
      Log.warn('OpenGL extension ARB_shader_objects not supported - shaders disabled')
      return
    if not glInitVertexShaderARB():
      Log.warn('OpenGL extension ARB_vertex_shader not supported - shaders disabled')
      return
    if not glInitFragmentShaderARB():
      Log.warn('OpenGL extension ARB_fragment_shader not supported - shaders disabled')
      return
    if not glInitMultitextureARB():
      Log.warn('OpenGL extension ARB_multitexture not supported - shaders disabled')
      return
    if not glInitTexture3DEXT():
      if sys.platform != 'darwin':
        Log.warn('OpenGL extension EXT_texture3D not supported - shaders disabled')
        return

    #stump: pyOpenGL 2.x compatibility
    try:
      checkFunctionsForPyOpenGL2x()
    except:
      Log.error('Shaders disabled due to pyOpenGL 2.x compatibility issue: ')
      return

    self.workdir = dir

    # Load textures needed by the shaders.
    try:
      self.noise3D = self.loadTex3D("noise3d.dds")
      self.outline = self.loadTex2D("outline.tga")
    except:
      Log.error('Could not load shader textures - shaders disabled: ')
      return

    self.multiTex = (GL_TEXTURE0_ARB,GL_TEXTURE1_ARB,GL_TEXTURE2_ARB,GL_TEXTURE3_ARB)
    self.enabled = True
    self.turnon = True

    # Compile the shader objects that we are going to use.
    # Also set uniform shader variables to default values.
    try:
      self.make("lightning","stage")
    except:
      Log.error("Error compiling lightning shader: ")
    else:
      self.enable("stage")
      self.setVar("ambientGlowHeightScale",6.0)
      self.setVar("color",(0.0,0.0,0.0,0.0))
      self.setVar("glowFallOff",0.024)
      self.setVar("height",0.44)
      self.setVar("sampleDist",0.0076)
      self.setVar("speed",1.86)
      self.setVar("vertNoise",0.78)
      self.setVar("fading",1.0)
      self.setVar("solofx",False)
      self.setVar("scalexy",(5.0,2.4))
      self.setVar("fixalpha",True)
      self.setVar("offset",(0.0,-2.5))
      self.disable()

    try:
      self.make("lightning","sololight")
    except:
      Log.error("Error compiling lightning shader: ")
    else:
      self.enable("sololight")
      self.setVar("scalexy",(5.0,1.0))
      self.setVar("ambientGlow",0.5)
      self.setVar("ambientGlowHeightScale",6.0)
      self.setVar("solofx",True)
      self.setVar("height",0.3)
      self.setVar("glowFallOff",0.024)
      self.setVar("sampleDist",0.0076)
      self.setVar("fading",4.0)
      self.setVar("speed",1.86)
      self.setVar("vertNoise",0.78)
      self.setVar("solofx",True)
      self.setVar("color",(0.0,0.0,0.0,0.0))
      self.setVar("fixalpha",True)
      self.setVar("glowStrength",100.0)  
      self.disable()

    try:
      self.make("lightning","tail")
    except:
      Log.error("Error compiling lightning shader: ")
    else:
      self.enable("tail")
      self.setVar("scalexy",(5.0,1.0))
      self.setVar("ambientGlow",0.1)
      self.setVar("ambientGlowHeightScale",6.0)
      self.setVar("solofx",True)
      self.setVar("fading",4.0)
      self.setVar("height",0.0)
      self.setVar("glowFallOff",0.024)
      self.setVar("sampleDist",0.0076)
      self.setVar("speed",1.86)
      self.setVar("vertNoise",0.78)
      self.setVar("solofx",True)
      self.setVar("color",(0.3,0.7,0.9,0.6))
      self.setVar("glowStrength",70.0) 
      self.setVar("fixalpha",True)
      self.setVar("offset",(0.0,0.0)) 
      self.disable()

    try:
      self.make("rockbandtail","tail2")
    except:
      Log.error("Error compiling rockbandtail shader: ")  
    else:
      self.enable("tail2")
      self.setVar("height",0.2)
      self.setVar("color",(0.0,0.6,1.0,1.0))
      self.setVar("speed",9.0)
      self.setVar("offset",(0.0,0.0))
      self.setVar("scalexy",(5.0,1.0))
      self.disable()

    try:
      self.make("metal","notes")
    except:
      Log.error("Error compiling metal shader: ")
    else:
      self.enable("notes")
      self.disable()

    try:
      self.make("neck","neck")
    except:
      Log.error("Error compiling neck shader: ")

    try:
      self.make("cd","cd")
    except:
      Log.error("Error compiling cd shader: ")

    #self.defineConfig()

def mixColors(c1,c2,blend=0.5):
  c1 = list(c1)
  c2 = list(c2)
  alpha = 0.0
  for i in range(3):
    c1[i] =  c1[i] + blend * c2[i]
    alpha += c1[i]
  c1 = c1[:3] + [min(alpha / 3.0,1.0)]
  return tuple(c1)

shaders = ShaderList()
del ShaderList
