import math
import numpy
import uuid

# base class for all objects that can be in a GridWorld. Not much here other than
# the world of which they are a part and their x-y coordinates in the world (which may
# be actual or believed coordinates)
class GridObject(object):

      def __init__(self, name, obj_id=None, world=None, x=0, y=0):

          # obscure: with a __setattr__ method override, setters for any sort of internal
          # attribute must be set even in the constructor using the base class' __setattr__ method.
          object.__setattr__(self,"_objectName",name) # what kind of object this is
          if obj_id is None:
             object.__setattr__(self,"_objectID",uuid.uuid4().hex) # unique identifier
          else:
             object.__setattr__(self,"_objectID",obj_id) # no special guards here against duplicate IDs - this is the user's responsibility!
          object.__setattr__(self,"_static",False) # agents which are not static can take actions.
          self.x = x
          self.y = y
          object.__setattr__(self,"_world",world)

      # name, object ID, and world cannot be set directly. name and ID are fixed at
      # at construction. world is set through the embed method below.
      def __setattr__(self,name,value):
          if name not in["_objectName","_objectID","_world","_static"]:
             object.__setattr__(self,name,value)

      @property
      def objectName(self):
          return self._objectName

      @property
      def objectID(self):
          return self._objectID

      @property
      def inWorld(self):
          return self._world

      def embed(self, world):
          self._world = world

      def place(self, world, x, y):
          if self._world is None:
             self._world = world
          if self._world == world:
             self.x = x
             self.y = y

class Action():

      # define the possible actions here
      inaction = -1
      move = 0
      tag = 1

      # set up a basic action. An action stores the agent, what action it is doing,
      # in what direction the action is made, any possible object of the action (self.actedUpon),
      # and the action start point position. 
      def __init__(self, agent, code, target, direction):

          self.agent = agent
          self.actionCode = code
          self.actionDirection = direction
          self.actedUpon = target 
          self.x = agent.x
          self.y = agent.y

# a GridTarget is a very simple object indeed, the only thing one
# can do to it is tag it.
class GridTarget(GridObject):

      def __init__(self, name, obj_id=None, world=None, x=0, y=0, seq=0):

          super().__init__("target", obj_id, world, x, y)
          self._tagged = False
          self._order = seq
          
      @property
      def isTagged(self):
          return self._tagged

      @property
      def sequenceNum(self):
          return self._order

      def tagTarget(self):
          self._tagged = True
          
            
class GridAgent(GridObject):

      # set up the agent, which needs a name, an ID, a world to live in, and a start point.
      def __init__(self, name, obj_id=None, world=None, x=0, y=0):

          # call the generic GridObject constructor to set up common properties
          super().__init__("agent", obj_id, world, x, y)
          # no current action selected
          self._currentAction = Action(self, Action.inaction, None, 0)
          self.owned = [] # any objects the agent may possess
          self._map = {} # a dictionary of (x,y) positions containing a target dictionary of accessible locations with distances
          self._frontier = [(self.x, self.y)] # initialise our start point
          self._backtrack = [] # this will keep track of what our path has been, so we can navigate back to a starting point
          self._goals = [] # what should the agent's goal(s) be? This can be set either internally or externally
          self._curPath = None # this will be the path list the agent will follow. None means nowhere to go; empty indicates at destination 

      # don't allow arbitrary redirection of current actions
      def __setattr__(self,name,value):
          if name != "_currentAction":
             GridObject.__setattr__(self,name,value)

      # actionResult gives the agent its observation model: what happens when it takes an action. In general, in the GridWorld,
      # the observation will be a returned object or location indicating the agent successfully acquired the object or occupied
      # the location. If there were objects this agent could have removed in its location, it can interrogate
      # the occupants property of a returned location to check that the object in concern no longer exists.
      def actionResult(self, result):
          # filter out non-actions
          if self._currentAction.actionCode > self._currentAction.inaction:
             # move action expects a GridPoint in return. Any result observations you add should check as below for
             # the correct class!
             if self._currentAction.actionCode == self._currentAction.move:
                if result is None:
                   return
                if result.__class__.__name__ != "GridPoint":
                   raise ValueError("Expected a GridPoint class for a Move action, got a {0} class instead".format(result.__class__.__name__))
                self.x = result.x
                self.y = result.y
             # any other actions you may implement should have their observed results dealt with here

             # tagging a target sets the current path to None (so we can plan another path)
             # and if the attempt to tag succeeded, removes the target from the goals. So
             # if the path reached the presumed destination but there was nothing to tag,
             # we must have gone astray, and a new path to the goal can be planned.
             if self._currentAction.actionCode == self._currentAction.tag:
                self._curPath = None
                if result is None:
                   return
                if result.__class__.__name__ != "GridTarget":
                   raise ValueError("Expected a GridTarget class for a Move action, got a {0} class instead".format(result.__class__.__name__))
                self._goals.pop(0)
          
      # this is the main function that generates intelligent behaviour. It implements
      # a 'policy': a mapping from the state (which you can get from the world, your x, y
      # position, and the occupants which you will get as a list), to an action.
      def chooseAction(self, world, x, y, occupants):

          # don't attempt to act in a world we're not in. This also prevents us from accidentally
          # resetting the world.
          if world != self._world:
             GridObject.__setattr__(self,"_currentAction",Action(self, Action.inaction, None,0))
             return self._currentAction

          # --- Insert your actions here ---

          # some goals still to reach?
          if len(self._goals) > 0:
             currentLoc = (self.x, self.y)
             # no path, so create a new one
             while self._curPath is None and len(self._goals) > 0:

                # TODO
                # ----- Choose which search method you are using with these lines -------
                   
                #self._curPath = self._iterativeDeepeningSearch(currentLoc,self._goals[0])
                #self._curPath = self._breadthFirstSearch(currentLoc,self._goals[0])
                self._curPath = self._AStarSearch(currentLoc, self._goals[0])

                
                # could have an unreachable goal, which we just remove
                if self._curPath is None:
                   self._goals.pop(0)
             # no goals left; everything remaining was unreachable, so do nothing.
             if len (self._goals) == 0:
                GridObject.__setattr__(self,"_currentAction",Action(self, Action.inaction, None,0))
                return self._currentAction
             # path includes our current location, which we can just pop, leaving the path as the
             # waypoints to the next goal
             if self._curPath[0] == currentLoc:
                self._curPath.pop(0)
             # at a goal point? Current path will be empty
             if len(self._curPath) == 0:
                if currentLoc == self._goals[0]:
                   objectToTag = None
                   try:
                       # is this a taggable goal? If so, tag it.
                       objectToTag = next(occ for occ in occupants if occ.objectName == "target")
                       GridObject.__setattr__(self, "_currentAction",Action(self, Action.tag, objectToTag,0))
                       return self._currentAction
                   except StopIteration:
                       pass
             else:
               # not at the goal point. Continue to move towards the next waypoint.
               GridObject.__setattr__(self,"_currentAction",Action(self,Action.move, None, self._getDirection(self._curPath[0])))
               return self._currentAction                                                             
          
          # default action is just a random move in some direction.
          GridObject.__setattr__(self,"_currentAction",Action(self, Action.move, None, round(numpy.random.uniform(-0.49999,3.5))))
          return self._currentAction

      # importMap updates the world map, either in whole or in part.
      def importMap(self, gridMap):
          self._map.update(gridMap)

      # addGoalPoint indicates that this x, y position is to be considered a goal point. Priority
      # allows the point to be inserted wherever desired in the list.
      def addGoalPoint(self, x, y, priority=-1):
          # some points may be reachable, but not in the map because they have been optimised
          # away. We can exploit the geometries of the GridWorld to derive these points, because
          # a valid goal MUST lie between 2 points which are in the map, which share either an
          # x or y coordinate, and which are connected
          if (x,y) not in self._map:
             # first, find the set of points that share an x or y coordinate with the goal
             alignedPoints = sorted([loc for loc in self._map if loc[0] == x or loc[1] == y])
             # now, see if one lies beyond the goal in the x or y direction
             nextInColumn = None
             nextInRow = None
             try:
                 # a Python generator function extracts the first point with the same
                 # x-coordinate lying below (i.e. has a larger y-value) than the goal.
                 nextInColumn = next(origin for origin in sorted(alignedPoints, key=lambda l: l[1]) if origin[1] > y)
             except StopIteration:
                 pass
             try:
                 # same idea for the a point with the same y coordinate lying to the right
                 # of the goal
                 nextInRow = next(origin2 for origin2 in alignedPoints if origin2[0] > x)
             except StopIteration:
                 pass
             # a column-aligned point was found. Does it have a connection (an edge) to a
             # neighbouring point lying above the goal?
             if nextInColumn is not None:
                previousInColumn = None
                try:
                    previousInColumn = next(dst for dst in sorted([loc3 for loc3 in self._map[nextInColumn]
                                                                   if loc3[0] == x and loc3[1] < y],
                                                                  key=lambda m: m[1], reverse=True))
                except StopIteration:
                    pass
                # goal is between 2 connected points in a column. Insert the goal point as a node,
                # and replace the single edge between the original 2 nodes with 2 edges linking the
                # goal point.
                if previousInColumn is not None:
                   self._map[(x,y)] = {nextInColumn: nextInColumn[1]-y, previousInColumn: y-previousInColumn[1]}
                   self._map[previousInColumn][(x,y)] = y-previousInColumn[1]
                   self._map[nextInColumn][(x,y)] = nextInColumn[1]-y
                   del self._map[previousInColumn][nextInColumn]
                   del self._map[nextInColumn][previousInColumn]
             # same logic, for row-aligned points
             if nextInRow is not None:
                previousInRow = None
                try:
                    previousInRow = next(dst2 for dst2 in sorted([loc4 for loc4 in self._map[nextInRow]
                                                                  if loc4[1] == y and loc4[0] < x],
                                                                 reverse=True))
                except StopIteration:
                    pass
                if previousInRow is not None:
                   self._map[(x,y)] = {nextInRow: nextInRow[0]-x, previousInRow: x-previousInRow[0]}
                   self._map[previousInRow][(x,y)] = x-previousInRow[0]
                   self._map[nextInRow][(x,y)] = nextInRow[0]-x
                   del self._map[previousInRow][nextInRow]
                   del self._map[nextInRow][previousInRow]
          # so now if the goal is still on the map, it is fundamentally unreachable.
          if (x,y) not in self._map:   
             raise ValueError("Can't get there from here! Specfied a goal point ({0},{1}) not reachable in agent {2}'s map".format(x, y, self.objectID))
             return
          # reachable points can be inserted at their appropriate point in the target list.
          if priority < 0:
             self._goals.append((x,y))
          else:
             self._goals.insert(priority,(x,y))

      # get rid of an existing goal point       
      def removeGoalPoint(self, x, y):
          try:
             self._goals.remove(x,y)
          except ValueError:
             pass
    
      # convenience function allows us to extract the direction to a target location
      def _getDirection(self, target):
          if target[0] == self.x:
             if target[1] == self.y:
                return self._world.Nowhere
             elif target[1] > self.y:
                return self._world.South
             else:
                return self._world.North
          elif target[0] < self.x:
             if target[1] != self.y:
                return self._world.Nowhere
             else:
                return self._world.West
          else:
             if target[1] != self.y:
                return self._world.Nowhere
             else:
                return self._world.East

      # an efficient way to identify if a tuple is in a list. Creates a python generator expression to evaluate. 
      def _inFrontier(self, target):
          try:
             nextTgt = next(loc for loc in self._frontier if loc[0] == target[0] and loc[1] == target[1])
          except StopIteration:
             return None
          return nextTgt

      # TODO
      # -------------- These are the methods you need to implement for the Week 2 practical --------

      # all these searches take a start point and a target, and return a path list ordered from start
      # to target, of the nodes the agent should traverse to reach the target

      # TODO
      # breadth-first search should expand each location completely before moving to the next. In the
      # gridworld, this isn't crippling, the branching factor is only 4, but consider how the problem
      # would scale to a 100*100 grid (!)
      def _breadthFirstSearch(self, start, target):
          if start not in self._map:
             return None 
          if start == target:
             return [start]
          explored = set()
          path = {}
          self._frontier = [start]
          while len(self._frontier) > 0:
                curNode = self._frontier.pop(0)
                explored.add(curNode)
                expansion = dict([(node,curNode) for node in self._map[curNode].keys()
                                  if node not in explored and node not in path])
                if target in expansion:
                   foundPath = [target]
                   parent = curNode
                   while parent in path:
                         foundPath.append(parent)
                         parent = path[parent]
                   foundPath.append(parent)
                   foundPath.reverse()
                   return foundPath
                else:
                   self._frontier.extend(expansion.keys())
                   path.update(expansion)
          return None

      # TODO
      # depth-first search expands each branch. This is most efficiently done recursively, and
      # by using a ply argument, we can trivially implement iterative deepening. An explored
      # parameter - a list of expanded nodes - makes sure we can't end up in endless loops
      def _depthFirstSearch(self, start, target, ply=0, explored=None):
          if start not in self._map:
             return None
          if start == target:
             return [start]
          if ply <= 0:
             return []
          if explored is None:
             explored = set([start])
          bottomedOut = False
          for nextNode in (node for node in self._map[start].keys() if node not in explored):
              foundPath = self._depthFirstSearch(nextNode, target, ply-1, explored.union(set([nextNode])))
              if foundPath is not None:
                 if len(foundPath) > 0:
                    foundPath.insert(0,start)
                    return foundPath
                 else:
                    bottomedOut = True
          if bottomedOut:
             return []
          return None
       
      # TODO           
      # here is the extension to iterative deepening
      def _iterativeDeepeningSearch(self, start, target):
          ply = 1
          foundPath = []
          while foundPath is not None and len(foundPath) == 0:
                foundPath = self._depthFirstSearch(start, target, ply)
                ply += 1
          return foundPath

      # TODO
      # A* search is an informed search, and expects a heuristic, which should be a
      # function of 2 variables, both tuples, the start, and the target.
      def _AStarSearch(self, start, target, heuristic=None):
          if start not in self._map:
             return None
          if start == target:
             return [start]
          if heuristic is None: heuristic = lambda x, y: math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
          # these are the nodes that have been completely expanded, so don't need to be traced backwards
          explored = set()
          # these are the nodes still to be explored, sorted by estimated cost. They need to have
          # the complete path stored because any one of them might contain the best solution. We
          # arrange this as a nested dictionary to get a reasonably straightforward way to look up
          # the cheapest path. A heapq could also work but introduces implementation complexities.
          expanded = {heuristic(start, target): {start: [start]}}
          while len(expanded) > 0:
                bestPath = min(expanded.keys())
                nextExpansion = expanded[bestPath]
                if target in nextExpansion:
                  return nextExpansion[target]
                nextNode = nextExpansion.popitem()
                while len(nextExpansion) > 0 and nextNode[0] in explored:
                      nextNode = nextExpansion.popitem()
                if len(nextExpansion) == 0:
                   del expanded[bestPath]
                if nextNode[0] not in explored:
                   explored.add(nextNode[0])
                   expansionTargets = [node for node in self._map[nextNode[0]].items() if node[0] not in explored]
                   while len(expansionTargets) > 0:
                         expTgt = expansionTargets.pop()
                         estimatedDistance = bestPath-heuristic(nextNode[0],target)+expTgt[1]+heuristic(expTgt[0],target)
                         if estimatedDistance in expanded:             
                            expanded[estimatedDistance][expTgt[0]] = nextNode[1]+[expTgt[0]]
                         else:
                            expanded[estimatedDistance] = {expTgt[0]: nextNode[1]+[expTgt[0]]}
          return None            
                 
                 
                                        
           
           
                
                         
                     
          
