
'''

    2019 CAB320 Sokoban assignment


'''

import search
import os

import sokoban
import math

from search import *
from sokoban import find_2D_iterator
from sokoban import Warehouse



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():

    return [(9935924, 'Greyden', 'Scott'), (9983244,'John', 'Santias'), (9918205,'Alex', 'Holm')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''
    Identifies the taboo cells of the warehouse. A cell inside the warehouse 
    is called 'taboo' if whenever a box get pushed on such a cell then the puzzle
    becomes unsolvable.
     Rule 1: if a cell is a corner inside the warehouse and not a target,
             then it is a taboo cell.
     Rule 2: all the cells between two corners inside the warehouse along a
             wall are taboo if none of these cells is a target.

    @param warehouse: a Warehouse object

    @return
       A string representing the puzzle with only the wall cells marked with
       an '#' and the taboo cells marked with an 'X'.
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.
    '''

    # Constants
    squares_to_remove = ['$', '@']
    target_squares = ['.', '!', '*']
    wall_square = '#'
    taboo_square = 'X'

    def is_corner(warehouse, x, y, wall=0):

        num_ud_walls = 0
        num_lr_walls = 0

        for (dx, dy) in [(0, 1), (0, -1)]:
            if warehouse[y + dy][x + dx] == wall_square:
                num_ud_walls += 1

        for (dx, dy) in [(1, 0), (-1, 0)]:
            if warehouse[y + dy][x + dx] == wall_square:
                num_lr_walls += 1
        if wall:
            return (num_ud_walls >= 1) or (num_lr_walls >= 1)
        else:
            return (num_ud_walls >= 1) and (num_lr_walls >= 1)

    warehouse_str = str(warehouse)

    for char in squares_to_remove:
        warehouse_str = warehouse_str.replace(char, ' ')

    warehouse_2d = [list(line) for line in warehouse_str.split('\n')]

    for y in range(len(warehouse_2d) - 1):
        inside = False
        for x in range(len(warehouse_2d[0]) - 1):

            if not inside:
                if warehouse_2d[y][x] == wall_square:
                    inside = True
            else:

                if all([cell == ' ' for cell in warehouse_2d[y][x:]]):
                    break
                if warehouse_2d[y][x] not in target_squares:
                    if warehouse_2d[y][x] != wall_square:
                        if is_corner(warehouse_2d, x, y):
                            warehouse_2d[y][x] = taboo_square

    for y in range(1, len(warehouse_2d) - 1):
        for x in range(1, len(warehouse_2d[0]) - 1):
            if warehouse_2d[y][x] == taboo_square and is_corner(warehouse_2d, x, y):
                row = warehouse_2d[y][x + 1:]
                col = [row[x] for row in warehouse_2d[y + 1:][:]]

                for x2 in range(len(row)):
                    if row[x2] in target_squares or row[x2] == wall_square:
                        break
                    if row[x2] == taboo_square and is_corner(warehouse_2d, x2 + x + 1, y):
                        if all([is_corner(warehouse_2d, x3, y, 1) for x3 in range(x + 1, x2 + x + 1)]):
                            for x4 in range(x + 1, x2 + x + 1):
                                warehouse_2d[y][x4] = 'X'

                for y2 in range(len(col)):
                    if col[y2] in target_squares or col[y2] == wall_square:
                        break
                    if col[y2] == taboo_square and is_corner(warehouse_2d, x, y2 + y + 1):
                        if all([is_corner(warehouse_2d, x, y3, 1) for y3 in range(y + 1, y2 + y + 1)]):
                            for y4 in range(y + 1, y2 + y + 1):
                                warehouse_2d[y4][x] = taboo_square

    warehouse_str = '\n'.join([''.join(line) for line in warehouse_2d])

    for char in target_squares:
        warehouse_str = warehouse_str.replace(char, ' ')
    return warehouse_str


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.
    When self.allow_taboo_push is set to True, the 'actions' function
    returns all possible legal moves including those that move a box on a taboo
    cell. If self.allow_taboo_push is set to False, those moves are not
    included in the returned list of actions.
    If self.macro is set True, the 'actions' function returns macro actions. 
    If self.macro is set False, the 'actions' function returnd elementary actions.
    '''

    def __init__(self, initial, goal):
        self.initial = (((-1,-1), "None"), initial)
        self.goal = goal.replace("@", " ")
        self.allow_taboo_push = True
        self.macro = True


    def actions(self, state):
        """
        Returns the list of actions that can be executed in the given state.
        """

        #Warehouse current state
        the_warehouse = sokoban.Warehouse()
        #Positional information of boxes, worker, targets and walls extracted
        the_warehouse.extract_locations(state[1].split(sep="\n"))
        #If allow_taboo_push is True, return all legal moves including the box on a taboo cell
        if self.allow_taboo_push:
            #Find bad cell spots
            is_cell_taboo = set(find_2D_iterator(taboo_cells(the_warehouse), "X"))
            #find directions for the box
            for box in the_warehouse.boxes:
                for offset in worker_offsets:
                    offset = offset_of_direction(offset)
                    b_position = add_coordinates(box, offset)
                    ppos1 = (box[0] + offset[0] * -1)
                    ppos0 = (box[1] + offset[1] * -1)
                    p_position = (ppos1, ppos0)
                    p_position = flip_coordinates(p_position)
                    if can_go_there(the_warehouse, p_position):
                        if b_position not in is_cell_taboo:
                            if b_position not in the_warehouse.boxes:
                                if b_position not in the_warehouse.walls:
                                    yield(box, direction_of_offset(offset))

        #if allow_taboo_push is False, taboo and shouldn't be included in list of moves.
        elif not self.allow_taboo_push:
            #find directions for the box
            for box in the_warehouse.boxes:
                for offset in worker_offsets:
                    b_position = add_coordinates(box, offset)
                    p_position = flip_coordinates((box[0] + offset[0] * -1), (box[1] + offset[1] * -1))
                    if can_go_there(the_warehouse, p_position):
                        if b_position not in the_warehouse.boxes:
                            if b_position not in the_warehouse.walls:
                                yield(box, direction_of_offset(offset))


    def result(self, state, move):
        '''
        Move is the direction of the object moved by the worker
        '''
        #Warehouse current state
        the_warehouse = sokoban.Warehouse()
        #Positional information of boxes, worker, targets and walls extracted
        the_warehouse.extract_locations(state[1].split(sep="\n"))
        #remove the box from its old position, set it to the character's offset direction
        #set the boxes' old position to the worker
        position = move[0]

        if position in the_warehouse.boxes:

            the_warehouse.worker = position
            the_warehouse.boxes.remove(position)
            offset_position = offset_of_direction(move[1])
            the_warehouse.boxes.append(add_coordinates(position, offset_position))
            return move, str(the_warehouse)
        else:
            raise ValueError("Box is outside the Warehouse")

# def check_each_action_and_move(warehouse, action_seq):
#     '''
#     Same purpose as check_action_seq function
#     NB: It does not check if it pushes a box onto a taboo cell.
    
#     @param warehouse: a Warehouse object
#     @param action_seq: a list of actions
    
#     @return
#         a altered warehouse
#     '''

#     for action in action_seq:
#         worker_x, worker_y = warehouse.worker
        
#         #Get localtion of two cells we should check
#         if action == 'Left':
#             cell1 = (worker_x-1, worker_y)
#             cell2 = (worker_x-2, worker_y)
            
#         elif action == 'Right':
#             cell1 = (worker_x+1, worker_y)
#             cell2 = (worker_x+2, worker_y)
        
#         elif action == 'Up':
#             cell1 = (worker_x, worker_y-1)
#             cell2 = (worker_x, worker_y-2)
        
#         elif action == 'Down':
#             cell1 = (worker_x, worker_y+1)
#             cell2 = (worker_x, worker_y+2)            
        
#         #Check whether the worker push walls
#         if cell1 in warehouse.walls:
#             return 'Failure'

#         if cell1 in warehouse.boxes:
#             if cell2 in warehouse.boxes or cell2 in warehouse.walls:
#                 #push two boxes or the box has already nearby the wall, faliure
#                 return 'Failure'
#             #Only push one box
#             warehouse.boxes.remove(cell1)
#             warehouse.boxes.append(cell2)

#         warehouse.worker = cell1 

#     return warehouse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_action_seq(warehouse, action_seq):
    '''

    Determine if the sequence of actions listed in 'action_seq' is legal or not.

    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']

    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return
               A string representing the state of the puzzle after applying
               the sequence of actions.
    '''

    for action in action_seq:
        x, y = warehouse.worker

        #Get location of two spaces to  check
        if action == 'Left':
            space_1 = (x - 1, y)
            space_2 = (x - 2, y)

        elif action == 'Right':
            space_1 = (x + 1, y)
            space_2 = (x + 2, y)

        elif action == 'Up':
            space_1 = (x, y - 1)
            space_2 = (x, y - 2)

        elif action == 'Down':
            space_1 = (x, y + 1)
            space_2 = (x, y + 2)

        #Check if the player has pushed walls
        if space_1 in warehouse.walls:
            return 'Failure'

        if space_1 in warehouse.boxes:
            if space_2 in warehouse.boxes or space_2 in warehouse.walls:
                #push two boxes or the box is already nearby the wall
                return 'Failure'

            #Only push one box
            warehouse.boxes.remove(space_1)
            warehouse.boxes.append(space_2)

        warehouse.worker = space_1

    return warehouse.__str__() if type(warehouse)!=str else warehouse


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    '''
    This function should solve using elementary actions
    the puzzle defined in a file.

    @param warehouse: a valid Warehouse object

    @return
        String returns 'Impossible' if puzzle cannot be solved
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, return []
    '''
    
    path = []

    #get macro actions
    macro_actions = solve_sokoban_macro(warehouse)
    
    if macro_actions == ['Impossible'] or len(macro_actions) == 0:
        return macro_actions
    
    #append the actions retrieved from the sokoban_macro definition
    for action in macro_actions:
        path.append(action[1])

    return path
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    '''
    Determines whether the worker can walk to the cell dst=(row,column)
    without pushing any box.

    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      Otherwise False
    '''
    dsta = (str(dst[0]), str(dst[1]))
    dst0 = dsta[0].replace(",","").replace("(","").replace(")","")
    dst1 = dsta[1].replace(",","").replace("(","").replace(")","")
    dst0 = int(dst0)
    dst1 = int(dst1)
    
    def heuristic(n):
        '''
        Determine the heuristic distance between the worker and the destination

        @param n: the node state "<Node (x, y)>"

        @return
          The heuristic distance sqrt(((x_worker - x_destination) ^2) + ((y_worker - y_destination) ^ 2))

        '''
        state = n.state

        dsta = (str(dst[0]), str(dst[1]))
        dst0 = dsta[0].replace(",","").replace("(","").replace(")","")
        dst1 = dsta[1].replace(",","").replace("(","").replace(")","")
        dst0 = int(dst0)
        dst1 = int(dst1)

        # distance = sqrt(xdiff^2 + ydiff^2). Basic distance formula heuristic.
        return math.sqrt(((state[1] - dst1) ** 2)
                         + ((state[0] - dst0) ** 2))

    # Destination (row,col), not (x,y)
    dst = (dst1, dst0)

    # A* graph search on the PathScanner search
    node = astar_graph_search(PathScanner(warehouse.worker, warehouse, dst), heuristic)
    print("NODE: ", node)

    #if there's a destination
    if node is not None:
        return True
    else:
        return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    """
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ]
    means that the worker first goes the box at row 3 and column 4 and pushes
    it left, then goes the box at row 5 and column 2 and pushes it up, and
    finally goes the box at row 12 and column 4 and pushes it down.

    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return ['Impossible']
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    """

    if warehouse.boxes == warehouse.targets:
        return []
    
    macroActions = SearchMacroActions(warehouse)

    #use A* graph search to move the box to the goal
    macroSolution = search.astar_graph_search(macroActions)
    
    final_macro_actions = macroActions.solution(macroSolution)
    return final_macro_actions
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - -  - - - - - - - Other classess and functions - - - - - - - - - - - -

# Worker's offsets (left, right, up and down) from its current position
worker_offsets = {'left':(-1, 0), 'right':(1, 0), 'up':(0, -1), 'down':(0, 1) }

class PathScanner(search.Problem):

    def __init__(self, initial, warehouse, goal):
        '''
        Assign the passed values

        @param
            initial: the initial value of the worker
            warehouse: the warehouse object
            goal: the destination
        '''
        self.initial = initial
        self.goal = goal
        self.warehouse = warehouse

    def result(self, state, nextMove):
        '''
        Apply the next move to the current state

        @param
            state: the current state
            nextMove: the worker's next move

        @return
            the next state
        '''
        nextState = add_coordinates(state, nextMove)
        return nextState

    def actions(self, state):
        '''
        Determine the next action for the worker using the offset values

        @param
            state: the current state of the worker

        @return
            the next possible position that isn't a wall or a box
        '''
        for worker_offset in worker_offsets.values():
            nextState = add_coordinates(state, worker_offset)
            if nextState not in self.warehouse.walls and nextState not in self.warehouse.boxes:
                yield worker_offset

class SearchMacroActions(search.Problem):

    def __init__(self, initial):
        '''
        Assign the passed values

        @param
            initial: the initial warehouse problem
        '''
        self.initial = initial
        self.present_boxes = []
        self.goal = initial.copy(boxes=initial.targets)
    
    def result(self, warehouse, action):
        '''
        The results of the macro actions

        @param
            warehouse: a valid Warehouse object
            action: list of actions to move boxes to the goals

        @return
            a warehouse object with boxes on the targets
        '''
        backup_warehouse = warehouse.copy(boxes=self.present_boxes.copy())

        #old position of the box
        old_pos = action[0]
        
        if old_pos in warehouse.boxes:
            warehouse.boxes.remove(old_pos)
        warehouse.worker = old_pos

        #move box to new location
        new_position = neighouring_cells(old_pos)
        pos = action_direction(action)

        #move box to new location
        warehouse.boxes.append(new_position[pos])

        return warehouse
    
    def actions(self, warehouse):    
        '''
        Finds all possible actions to move the boxes to the target

        @param
            warehouse: a valid Warehouse object

        @return
            list of possible moves
        '''
        potential_moves  = []
        #backup current boxes location
        self.present_boxes = warehouse.boxes.copy()
        deadlocks = deadlock_cells(warehouse)
        
        #the pushable boxes with direction and the worker nearby 
        pushable_boxes, worker_near_box = self.can_push_boxes(warehouse.copy())
        
        for box in pushable_boxes:
            #the worker's location around the boxes
            location_around_box = set(worker_near_box) & set(neighouring_cells(box).values())
            
            for worker in location_around_box:
                worker_offsets = worker[0] - box[0], worker[1] - box[1]
                
                #Check the boxes should be push or not in the next cell
                next_cell = box[0] - worker_offsets[0], box[1] - worker_offsets[1]
                if next_cell not in deadlocks \
                    and next_cell not in warehouse.walls \
                    and next_cell not in warehouse.boxes:
                    #the second cell is not deadlocks/boxes/walls, so it can be pushed
                    if worker_offsets == (0, 1):
                        potential_moves.append((box, "Up"))
                    elif worker_offsets == (0, -1):
                        potential_moves.append((box, "Down"))
                    elif worker_offsets == (1, 0):
                        potential_moves.append((box, "Left"))
                    elif worker_offsets == (-1, 0):
                        potential_moves.append((box, "Right"))
                    
        return potential_moves

    def h(self, n):
        '''
        Returns the heuristic value of the given node n

        @param
            n: the node

        @return
            heuristic value
        '''
        current_heuristic = 0
        for box in n.state.boxes:
            nearest_to_target = n.state.targets[0]
            for target in n.state.targets:
                manhattan_target = manhattan_distance(target, box)
                manhattan_closest = manhattan_distance(nearest_to_target, box)
                if (manhattan_target < manhattan_closest):
                    nearest_to_target = target
                    
            current_heuristic = current_heuristic + manhattan_distance(nearest_to_target, box)         
    
        return current_heuristic
    
    def can_push_boxes(self, warehouse):
        '''
        Finds all the boxes that can be pushed by the worker

        @param
            warehouse: a valid Warehouse object

        @return
            A tuple of sets containing the boxes that can be pushed and 
            the worker's location near the boxes
        '''
        pushable_boxes = set() #boxes_can_be_pushed
        near_boxes = set() #worker_locations_nearby_boxes
        unworkableCells = deadlock_cells(warehouse) #dead_locks
        workableCells = workable_cells(warehouse) #valid_cells

        #check all of cells worker can reach
        for workableCell in workableCells:
            #check any box can move to neighbour cell
            boxes_can_push_temp = set(warehouse.boxes) & set(neighouring_cells(workableCell).values())
            #check if worker can reach the cell and the box nearby the cell can be pushed
            if can_go_there(warehouse, (workableCell[1], workableCell[0])) and boxes_can_push_temp != set():
                # worker can go to this cell which is nearby one box
                for temp_box in boxes_can_push_temp:
                    #check each possible pushable boxes nearby the worker
                    offset = temp_box[0]-workableCell[0], temp_box[1]-workableCell[1]
                    second_cell = temp_box[0]+offset[0], temp_box[1]+offset[1]
                    if second_cell not in unworkableCells \
                    and second_cell not in warehouse.walls \
                    and second_cell not in warehouse.boxes:
                        near_boxes.add(workableCell)
                        pushable_boxes.add(temp_box)
                
        return (pushable_boxes, near_boxes)

    def goal_test(self, warehouse):
        '''
        Tests if the boxes are on the targets

        @param
            warehouse: a valid Warehouse object

        @return
            True if the boxes are on the targets. 
            Otherwise False
        '''
        return warehouse.boxes == self.goal.boxes
    
    def solution(self, targetNode):
        '''
        Finds the actions to move the box to the target positions

        @param
            warehouse: a valid Warehouse object
            action: list of actions to move boxes to the goals

        @return
            a list of actions containing coordinates and directions
            e.g. [((1, 2), 'Right), ((2, 2), 'right)]
        '''
        if targetNode == None:
            return ['Impossible']
        
        solution = [] #stores node actions
        final_solution = [] #final list with correct
        
        path = targetNode.path()
        
        for node in path:
            if node is not None:
                solution.append(node.action)
    
        #remove all None values in list
        solution.remove(None)

        #append the current position
        final_solution.append(( (solution[0][0][1], solution[0][0][0]), solution[0][1]))
        
        for action in solution:
            if action is not None:
                if action[1] =='Right':
                    final_solution.append(((action[0][1], action[0][0] + 1), action[1]))
                elif action[1] =='Left':
                    final_solution.append(((action[0][1], action[0][0] - 1), action[1]))
                elif action[1] =='Up':
                    final_solution.append(((action[0][1] - 1, action[0][0]), action[1]))
                elif action[1] =='Down':
                    final_solution.append(((action[0][1] + 1, action[0][0]), action[1]))
    
        return final_solution

def find_goal_coordinates(box, direction):
    '''
        FInds the offset values of the box to the goal its moving towards

        @param
            box: the location of the box
            direction: the move direction of the box

        @return
            the offset value
        '''
    if direction == "Up":
        offset = (0, 1)
    elif direction == "Down":
        offset = (0, -1)
    elif direction == "Left":
        offset = (1, 0)
    elif direction == "Right":
        offset = (-1, 0)
    return add_coordinates(box, offset)


def manhattan_distance(a, b):
    '''
    Calculates the manhattan distance between point a and point b

    @param
        a: the starting position
        b: the final position

    @return
        the distance to the final position
    '''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def flip_coordinates(c):
    '''
        flips tuples.
        Used for actions obtained from nodes where values 
        are in wrong order

        @param
            c: the given coordinates

        @return
            Flipped tuple. e.g (2, 1) becomes (1, 2)
        '''
    return (c[1], c[0])

def add_coordinates(c1, c2):
    '''
        Adds coordinates together for moving towards an object or target

        @param
            c1: starting position
            c2: target position

        @return
            Coordinates
        '''
    c = (c1[0] + c2[0], c1[1] + c2[1])
    return c

def offset_of_direction(direction):
    '''
    Finds the offset values of the given direction

    @param
        direction: "up", "down", "left", "right"

    @return
        the offset value. E.g. (0, 1), (0, -1) ....
    '''
    if offset == "up":
        return (0, -1)
    elif offset == "down":
        return (0, 1)
    elif offset == "left":
        return (-1, 0)
    elif offset == "right":
        return (1, 0)
    else:
        raise ValueError("Invalid direction")


def direction_of_offset(offset):
    '''
        Find the direction of the given offset

        @param
            offset: values (0, 1), (0, -1) etc...

        @return
            the direction. E.g. "up", "down", "left", "right"
        '''
    if offset == (0, -1):
        return "up"
    elif offset == (0, 1):
        return "down"
    elif offset == (-1, 0):
        return "left"
    elif offset == (1, 0):
        return "right"
    else:
        raise ValueError("Invalid offset")

def neighouring_cells(position):
    '''
        Find the neighbour positions/cells of the current spot

        @param
            position: the current position in x, y. (0, 1), (2, 1)...

        @return
            the neighbouring cells in the positions left, right, above 
            and below the current position
        '''
    x_position, y_position = position
    neighbours = { 'up':(x_position, y_position- + 1), 'down':(x_position, y_position - 1), 
                            'left':(x_position - 1, y_position), 'right':(x_position + 1, y_position) }
    return neighbours

def action_direction(action):
    '''
        Obtain the direction of the action

        @param
            action: Move found from searching nodes. E.g. ((1, 3), 'Right')

        @return
            The direction "Up", "Down", "left" or "right
        '''
    if action[1] == "Up":
        direction = "down"
    elif action[1] == "Down":
        direction = "up"
    elif action[1] == "Right":
        direction = "right"
    elif action[1] == "Left":
        direction = "left"
    
    return direction

def taboo_cells_corner(warehouse, workableCells):
    '''
        Finds the taboo cells located in each corner of the warehouse

        @param
            warehouse: a valid warehouse object
            workableCells: cells inside the warehouse where worker or boxes can move to

        @return
            A set of taboo cells in each corner of the warehouse
        '''
    taboo_cells = set([cell for cell in workableCells
    if((neighouring_cells(cell)['up'] in warehouse.walls or neighouring_cells(cell)['down'] in warehouse.walls) and
        (neighouring_cells(cell)['left'] in warehouse.walls or neighouring_cells(cell)['right'] in warehouse.walls))])
    return taboo_cells

def workable_cells(warehouse):
    '''
        Cells inside the warehouse where worker or boxes can move to. Excluding the walls

        @param
            warehouse: a valid warehouse object

        @return
            A set of cells where objects are able to move to.
        '''
    frontier = set()
    explored_cells = set()
    frontier.add(warehouse.worker) 

    while frontier:
        current_position = frontier.pop()
        explored_cells.add(current_position)

        neighbour_cells = neighouring_cells(current_position)
       
        for neighbour_cell in neighbour_cells.values():
            if (neighbour_cell not in frontier 
                and neighbour_cell not in explored_cells
                and neighbour_cell not in warehouse.walls):
                frontier.add(neighbour_cell)
    return explored_cells

def deadlock_cells(warehouse):
    '''
        Finds all the deadlocks inside the warehouse

        @param
            warehouse: a valid warehouse object

        @return
            A set of all the deadlocks inside the warehouse
        '''

    free_cells = workable_cells(warehouse)

    #target cells considered deadlock cells
    for target in warehouse.targets:
        free_cells.discard(target)
    
    #Corner deadlocks
    corner_deadlocks = taboo_cells_corner(warehouse, free_cells)
    
    # Deadlocks along the walls
    deadlock_alongWall = set()

    final_deadlocks = itertools.combinations(corner_deadlocks, 2)
    for cell_a, cell_b in final_deadlocks: 
        wallOrTarget = False
        x1, y1 = cell_a[0], cell_a[1]
        x2, y2 = cell_b[0], cell_b[1]

        #swap values because coordinates given is in wrong order
        if x1 > x2:
            swap = x1
            x1 = x2
            x2 = swap
        if y1 > y2:
            swap = y1
            y1 = y2
            y2 = swap

        if x1 == x2:
            ## target or wall between
            for y in range(y1 + 1, y2):
                if (x1, y) in warehouse.walls or (x1, y) in warehouse.targets:
                    wallOrTarget = True
                    break
            if wallOrTarget:
                continue
            
            ##Check if along the wall
            left = [False for y in range(y1, y2+1) if (x1 - 1, y) not in warehouse.walls]
            right = [False for y in range(y1, y2+1) if (x1 + 1, y) not in warehouse.walls]
            wallIsLeft = not False in left
            wallIsRight = not False in right

            # append all deadlock cells along the wall into set
            if wallIsLeft or wallIsRight:
                deadlock_alongWall |=  set([(x1, y) for y in range(y1+1, y2)])
        
        if y1 == y2:            
            ## target or wall between
            for x in range(x1+1,x2):
                if (x,y1) in warehouse.walls or (x,y1) in warehouse.targets:
                    wallOrTarget = True
                    break
            if wallOrTarget:
                continue
            
            ##Check if along the wall                           
            up = [False for x in range(x1, x2+1) if (x+1, y1-1) not in warehouse.walls]
            down = [False for x in range(x1, x2+1) if (x, y1+1) not in warehouse.walls]
            wallIsAbove = not False in up
            wallIsBelow = not False in down
            
            # append all deadlock cells along the wall into set
            if wallIsAbove or wallIsBelow:
                deadlock_alongWall |= set([(x,y1) for x in range(x1+1, x2)])      
    
    # Combine sets into one
    corner_deadlocks |= deadlock_alongWall

    return corner_deadlocks

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - -  - - - - - - -  - - - - - - Main - - - - - - - - - - - - - - - - - -

START_WAREHOUSE = 1
END_WAREHOUSE = 205

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: creating directory ' + directory)

def warehouse_solution(number):
    result = None
    average_time = 0
    problem_file = "./warehouses/warehouse"
    warehouse_problem = "./warehouses/warehouse_{:02d}.txt".format(number)
    wh = Warehouse()
    wh.load_warehouse(warehouse_problem)
    solution = solve_sokoban_elem(wh)
    print(solution)

    with open("./Warehouse_solutions/warehouse_{:02d}.txt".format(number), "w+") as file:
        file.write("The solution for warehouse {} is {}".format(number, solution))
        file.close()

if __name__ == "__main__":
    createFolder('./Warehouse_solutions')
    for i in range(START_WAREHOUSE, END_WAREHOUSE, 2):
        warehouse_solution(i)