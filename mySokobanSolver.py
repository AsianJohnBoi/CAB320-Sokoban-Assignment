
'''

    2019 CAB320 Sokoban assignment

The functions and classes defined in this module will be called by a marker script.
You should complete the functions and classes according to their specified interfaces.

You are not allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the
interface and triggers to a fail for the test of your code.

# by default does not allow push of boxes on taboo cells
SokobanPuzzle.allow_taboo_push = False

# use elementary actions if self.macro == False
SokobanPuzzle.macro = False

'''

# you have to use the 'search.py' file provided
# as your code will be tested with this specific file
import search

import sokoban
import math

from search import *
from sokoban import find_2D_iterator



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''

    return [(9935924, 'Greyden', 'Scott'), (9935924,'John', 'Santias'), (9918205,'Alex', 'Holm')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''
    Identify the taboo cells of a warehouse. A cell inside a warehouse is
    called 'taboo' if whenever a box get pushed on such a cell then the puzzle
    becomes unsolvable.
    When determining the taboo cells, you must ignore all the existing boxes,
    simply consider the walls and the target cells.
    Use only the following two rules to determine the taboo cells;
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

    Your implementation should be fully compatible with the search functions of
    the provided module 'search.py'.

    Each instance should have at least the following attributes
    - self.allow_taboo_push
    - self.macro

    When self.allow_taboo_push is set to True, the 'actions' function should
    return all possible legal moves including those that move a box on a taboo
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.

    If self.macro is set True, the 'actions' function should return
    macro actions. If self.macro is set False, the 'actions' function should
    return elementary actions.
    '''

    def __init__(self, initial, goal):
        self.initial = (((-1,-1), "None"), initial)
        self.goal = goal.replace("@", " ")
        self.allow_taboo_push = True
        self.macro = True


    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.

        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
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

        #if macro is true return macro actions
        #if self.macro:
        #    return

        # if macro is false use elementary actions
        #elif not self.macro:
        #    raise NotImplementedError

        #raise NotImplementedError

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_action_seq(warehouse, action_seq):
    '''

    Determine if the sequence of actions listed in 'action_seq' is legal or not.

    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.

    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']

    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
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
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''

    sokoban_macro_actions = solve_sokoban_macro(warehouse)

    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    '''
    Determine whether the worker can walk to the cell dst=(row,column)
    without pushing any box.

    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
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

    dst = (dst1, dst0)  # Destination is given in (row,col), not (x,y)

    # Use an A* graph search on the FindPathProblem search

    node = astar_graph_search(PathScanner(warehouse.worker, warehouse, dst), heuristic)
    print("NODE: ", node)

    # If a node was found, this is a valid destination
    if node is not None:
        return True
    else:
        return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#restructured
def solve_sokoban_macro(warehouse):
    #returns [] if the boxes are on the targets
    if warehouse.boxes == warehouse.targets:
        return []
    
    macroActions = SearchMacroActions(warehouse)

    #use A* graph search to move the box to the goal
    macroSolution = search.astar_graph_search(macroActions)
    
    final_macro_actions = macroActions.solution(macroSolution)
    
    return final_macro_actions


#restructured
def workable_cells(warehouse):
    frontier = set()
    explored_cells = set()
    frontier.add(warehouse.worker) #adds worker's position

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

#new definition
def taboo_cells_corner(warehouse, workableCells):
    taboo_cells = set([cell for cell in workableCells
    if((neighouring_cells(cell)['up'] in warehouse.walls or neighouring_cells(cell)['down'] in warehouse.walls) and
        (neighouring_cells(cell)['left'] in warehouse.walls or neighouring_cells(cell)['right'] in warehouse.walls))])
    return taboo_cells

def deadlock_cells(warehouse):

    free_cells = workable_cells(warehouse)

    #Remove all of target cells, which do need to be consider deadlock case
    for target in warehouse.targets:
        free_cells.discard(target)
    
    #Get the corner deadlock
    corner_deadlocks = taboo_cells_corner(warehouse, free_cells)
    
    # Get the deadlock along the walls
    deadlock_alongWall = set()

    final_deadlocks = itertools.combinations(corner_deadlocks, 2)
    for cell_a, cell_b in final_deadlocks: 
        wallOrTarget = False
        x1, y1 = cell_a[0], cell_a[1]
        x2, y2 = cell_b[0], cell_b[1]
        if x1 == x2:
            ## check whether there is a target or wall between them
            for y in range(y1 + 1, y2):
                if (x1, y) in warehouse.walls or (x1, y) in warehouse.targets:
                    wallOrTarget = True
                    break
            if y1 > y2:
                y1, y2 = y2, y1
            if wallOrTarget:
                continue
            
            ##check whether they are along the wall 
            up = [False for x in range(x1, x2+1) if (x1+1, y-1) not in warehouse.walls]
            down = [False for x in range(x1, x2+1) if (x1, y+1) not in warehouse.walls]
            wallIsAbove = not False in up
            wallIsBelow = not False in down

            # append all deadlock cells along the wall into set
            if alongWall_left or alongWall_right:
                deadlock_alongWall |=  set([(x1, y) for y in range(y1+1, y2)])
        
        if y1 == y2:
            ## check whether there is target between them
            for x in range(x1+1,x2):
                if (x,y1) in warehouse.walls or (x,y1) in warehouse.targets:
                    wallOrTarget = True
                    break
            if x1 > x2:
                x1, x2 = x2, x1
            if wallOrTarget:
                continue
            
            ##check whether they are along the wall                            
            up = [False for x in range(x1, x2+1) if (x1+1, y-1) not in warehouse.walls]
            down = [False for x in range(x1, x2+1) if (x1, y+1) not in warehouse.walls]
            wallIsAbove = not False in up
            wallIsBelow = not False in down
            
            # append all deadlock cells along the wall into set
            if alongWall_top or alongWall_bottom:
                deadlock_alongWall |= set([(x,y1) for x in range(x1+1, x2)])      
    
    # Merge all of deadlock to the single set
    corner_deadlocks |= deadlock_alongWall

    return corner_deadlocks    

#new definition
def action_direction(action):

    if action[1] == "Up":
        direction = "down"
    elif action[1] == "Down":
        direction = "up"
    elif action[1] == "Right":
        direction = "right"
    elif action[1] == "Left":
        direction = "left"
    
    return direction

class SearchMacroActions(search.Problem):

    def __init__(self, initial):
        self.initial = initial
        self.present_boxes = []
        self.goal = initial.copy(boxes=initial.targets)
    
    #restructured
    def result(self, warehouse, action):
        backup_warehouse = self.present_boxes.copy()
        
        old_pos = action[0] #old position of the box
        
        warehouse.boxes.remove(old_pos)
        warehouse.worker = old_pos
        #move box to new location
        new_position = neighouring_cells(old_pos)
        pos = action_direction(action)
        #move box to new location
        warehouse.boxes.append(new_position[pos])

        return warehouse
    
    #restructured
    def actions(self, warehouse):       
        potential_moves  = []
        #backup current boxes location
        self.present_boxes = warehouse.boxes.copy()
        deadlocks = deadlock_cells(warehouse)
        
        #get all of pushable boxes with direction, and worker locations which nearby them
        pushable_boxes, worker_near_box = self.can_push_boxes(warehouse.copy())
        
        #check if each action is valid or not
        for box in pushable_boxes:
            #get the worker location which around boxes
            location_around_box = set(worker_near_box) & set(neighouring_cells(box).values())
            
            for worker in location_around_box:
                worker_offsets = worker[0]-box[0], worker[1]-box[1]
                
                #get the second cell to check if the boxes should be pushed or not
                next_cell = box[0]-worker_offsets[0], box[1]-worker_offsets[1]
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
    
    #restructured
    def goal_test(self, warehouse):
        return warehouse.boxes == self.goal.boxes

    #restructured
    def h(self, n):
        current_heuristic = 0
        for box in n.state.boxes:
            #Find closest target
            nearest_to_target = n.state.targets[0]
            for target in n.state.targets:
                manhattan_target = manhattan_distance(target, box)
                manhattan_closest = manhattan_distance(nearest_to_target, box)
                if (manhattan_target < manhattan_closest):
                    nearest_to_target = target
                    
            #Update Heuristic
            current_heuristic = current_heuristic + manhattan_distance(nearest_to_target, box)         
    
        return current_heuristic
    
    #restructured
    def can_push_boxes(self, warehouse):
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
    
    #restructured
    def solution(self, targetNode):
        if targetNode == None:
            return ['Impossible']
        
        solution = [] #stores node actions
        final_solution = [] #final list with correct
        
        path = targetNode.path()
        
        for node in path:
            if node is not None:
                solution.append(node.action)
        
        for action in solution:
            if action is not None:
                final_solution.append(((action[0][1], action[0][0]), action[1]))
    
        return final_solution

    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - -  - - - - - - - Other classess and functions - - - - - - - - - - - -
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

# Worker's offsets (left, right, up and down) from its current position
worker_offsets = {'left':(-1, 0), 'right':(1, 0), 'up':(0, -1), 'down':(0, 1) }

def get_coordinates(warehouse):
    #Seperated characters appended to list
    warehouse_list = str(warehouse).split('\n')


    #Seperated characters appended to list
    data = []
    for each in str(warehouse):
        data.append(each)

    #counts the number of elements before creating a new line
    count = 0
    for each in warehouse_list:
        if each != '\n':
            count += 1
        else:
            break

    #removes '\n' in the list
    for i in data:
        if i == '\n':
            data.remove(i)

    #creates a list of coordinates (x,y)
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    theCoordinates = (list(chunks(data, count)))

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def flip_coordinates(c):
    return (c[1], c[0])

def add_coordinates(c1, c2):
    c = (c1[0] + c2[0], c1[1] + c2[1])
    return c

def offset_of_direction(offset):
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
    x_position, y_position = position
    neighbours = { 'up':(x_position, y_position- + 1), 'down':(x_position, y_position - 1), 
                            'left':(x_position - 1, y_position), 'right':(x_position + 1, y_position) }
    return neighbours
