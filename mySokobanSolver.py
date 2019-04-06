
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
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [(9935924, 'Greyden', 'Scott'), (9935924,'John', 'Santias'), (9935924,'Alex', 'Holm')]

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
                                warehouse_2d[y4][x] = 'X'

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
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' function is needed
    #     to satisfy the interface of 'search.Problem'.

    
    def __init__(self, warehouse):
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        raise NotImplementedError

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
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()


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
    
    ##         "INSERT YOUR CODE HERE"
    
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
    def heuristic(n):
        '''
        Determine the heuristic distance between the worker and the destination

        @param n: the node state (x, y)

        @return
          The heuristic distance sqrt(((x_worker - x_destination) ^2) + ((y_worker - y_destination) ^ 2))

        '''
        return math.sqrt(((n.state[0] - dst[0]) ** 2) + ((n.state[1] - dst[1]) ** 2)) 

    # A* graph search used on the NextPath search
    node = astar_graph_search(NextPath(warehouse.worker, warehouse, (dst[1], dst[0])), heuristic)

    # If found a node, return True otherwise False
    return True if node is not None else False

# Worker's offsets. Left, right, up and down
worker_offsets = {'left':(-1, 0), 'right':(1, 0), 'up':(0, -1), 'down':(0, 1) } 

class NextPath(search.Problem):
    def __init__(self, initial, warehouse, goal=None):
        '''
        Assign the passed values

        @param
            initial: the initial value of the worker
            warehouse: the warehouse object
            goal: the destination
        '''
        self.initial = initial
        self.warehouse = warehouse
        self.goal = goal

    def result(self, state, nextMove):
        '''
        Apply the next move to the current state

        @param 
            state: the current state
            nextMove: the worker's next move

        @return
            the next state
        '''
        nextState = state[0] + nextMove[0], state[1] + nextMove[1]
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
            nextState = state[0] + worker_offset[0], state[1] + worker_offset[1]
            if nextState not in self.warehouse.walls and nextState not in self.warehouse.boxes:
                yield worker_offset
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

