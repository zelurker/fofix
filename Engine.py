#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2006 Sami Ky�stil�                                  #
#               2008 evilynux <evilynux@gmail.com>                  #
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

import gc

import Network
import Object
from World import World
from Task import Task
import pygame
import Player

class Engine:
  """Main task scheduler."""
  def __init__(self, fps = 60):
    self.tasks = []
    self.frameTasks = []
    self.fps = fps
    self.currentTask = None
    self.paused = []
    self.running = True
    self.clock = pygame.time.Clock()

  def quit(self):
    for t in list(self.tasks + self.frameTasks):
      self.removeTask(t)
    self.running = False

  def addTask(self, task, synchronized = True):
    """
    Add a task to the engine.
    
    @param task:          L{Task} to add
    @type  synchronized:  bool
    @param synchronized:  If True, the task will be run with small
                          timesteps tied to the engine clock.
                          Otherwise the task will be run once per frame.
    """
    if synchronized:
      queue = self.tasks
    else:
      queue = self.frameTasks
      
    if not task in queue:
      queue.append(task)
      task.started()

  def removeTask(self, task):
    """
    Remove a task from the engine.
    
    @param task:    L{Task} to remove
    """
    found = False
    queues = self._getTaskQueues(task)
    for q in queues:
      q.remove(task)
    if queues:
      task.stopped()

  def _getTaskQueues(self, task):
    queues = []
    for queue in [self.tasks, self.frameTasks]:
      if task in queue:
        queues.append(queue)
    return queues

  def pauseTask(self, task):
    """
    Pause a task.
    
    @param task:  L{Task} to pause
    """
    self.paused.append(task)

  def resumeTask(self, task):
    """
    Resume a paused task.
    
    @param task:  L{Task} to resume
    """
    self.paused.remove(task)
    
  def enableGarbageCollection(self, enabled):
    """
    Enable or disable garbage collection whenever a random garbage
    collection run would be undesirable. Disabling the garbage collector
    has the unfortunate side-effect that your memory usage will skyrocket.
    """
    if enabled:
      gc.enable()
    else:
      gc.disable()
      
  def collectGarbage(self):
    """
    Run a garbage collection run.
    """
    gc.collect()

  def _runTask(self, task, ticks = 0):
    if not task in self.paused:
      self.currentTask = task
      task.run(ticks)
      self.currentTask = None

  def run(self):
    """Run one cycle of the task scheduler engine."""
    if not self.frameTasks and not self.tasks:
      return False
    
    for task in self.frameTasks:
      self._runTask(task)
    tick = self.clock.get_time()
    for task in self.tasks:
      self._runTask(task, tick)
    self.clock.tick(self.fps)
    return True
