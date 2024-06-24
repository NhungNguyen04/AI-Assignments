import math
import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # beginBox là vị ban đầu của các box
    beginPlayer = PosOfPlayer(gameState) # beginPlayer là vị trí ban đầu của người chơi

    # Mỗi node trong thuật toán này là tập hợp các state (bao gồm vị trí của player và box)
    # từ lúc bắt đầu trò chơi đến hiện tại (tức là một đường đi từ lúc bắt đầu đến hiện tại)
    # được tổ chức theo mảng với thứ tự FIFO. Khi xét một node thì các node con của nó sẽ được
    # add vào cuối để xét ở lượt tiếp theo.
    startState = (beginPlayer, beginBox) # node đầu tiên của thuật toán
    frontier = collections.deque([[startState]]) # Frontier là mảng các node, với node đầu tiên là startState
    exploredSet = set() # exploredSet là các node đã xét, khởi tạo rỗng vì trò chơi chưa bắt đầu
    actions = collections.deque([[0]]) # actions là tập hợp các nước đi hợp lệ của từ các node
    temp = [] # lưu trữ nước đi cuối cùng trước khi kết thúc ván chơi

    while frontier: # Tiếp tục vòng lặp nếu còn node chưa xét tới hoặc đã tìm được lời giải
        node = frontier.popleft() # Xét node đầu tiên trong mảng, ứng với thứ tự FIFO trong BFS
        node_action = actions.popleft() # Xét nước đi đầu tiên từ node hiện tại, ứng với FIFO
        if isEndState(node[-1][-1]):
            temp += node_action[1:] # Nếu vị trí của box nằm ở đích thì kết thúc trò chơi và trả về các nước đã đi
            break
        if node[-1] not in exploredSet: # Nếu node hiện tại chưa xét thì xét, nếu rồi thì xét node khác
            exploredSet.add(node[-1])  # add node hiện tại vào danh sách các node đã xét
            for action in legalActions(node[-1][0], node[-1][1]): # xét các nước đi hợp lệ từ node hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # tính toán trạng thái sau khi đi một nước
                if isFailed(newPosBox): # Nếu nước này dẫn box tới trạng thái fail (bị kẹt và không thể di chuyển được) thì bỏ qua
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Nếu không thì cập nhật node và add vào cuối mảng
                actions.append(node_action + [action[-1]]) # cập nhật các nước đã đi và add vào cuối mảng

    return temp # Khi tìm được lời giải thì trả về các nước đã đi


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""

    # Mỗi node trong thuật toán này là tập hợp các state (bao gồm vị trí của player và box)
    # từ lúc bắt đầu trò chơi đến hiện tại (tức là một đường đi từ lúc bắt đầu đến hiện tại),
    # được tổ chức trong một mảng với độ ưu tiên ứng với chiều dài quãng đường (chiều dài càng ngắn) thì độ ưu tiên càng lớn
    beginBox = PosOfBoxes(gameState) # beginBox là vị ban đầu của các box
    beginPlayer = PosOfPlayer(gameState) # beginPlayer là vị trí ban đầu của người chơi
    startState = (beginPlayer, beginBox) # node đầu tiên của thuật toán
    frontier = PriorityQueue()  # Frontier là mảng các node, mỗi node có một độ ưu tiên (priority) khác nhau
    frontier.push([startState], 0) # Node đầu tiên là startSate với độ ưu tiên 0
    exploredSet = set()  # exploredSet là các node đã xét, khởi tạo rỗng vì trò chơi chưa bắt đầu
    actions = PriorityQueue() # actions là tập hợp các nước đi hợp lệ của từ các node với độ ưu tiên khác nhau
    actions.push([0], 0) # nước đi đầu tiên chưa có, với độ ưu tiên 0
    temp = [] # lưu trữ các nước đã đi trong ván chơi
    ### CODING FROM HERE ###
    while frontier:
        node = frontier.pop() # Xét node với độ ưu tiên cao nhất
        node_action = actions.pop() # Xét nước đi với độ ưu tiên cao nhất từ node hiện tại

        if isEndState(node[-1][-1]):
            temp += node_action[1:] # Nếu vị trí của box nằm ở đích thì kết thúc trò chơi và trả về nước đi gần nhất đã dẫn đến đích
            break
        if node[-1] not in exploredSet: # Nếu node hiện tại chưa xét thì xét, nếu rồi thì xét node khác
            exploredSet.add(node[-1])  # add node hiện tại vào danh sách các node đã xét
            for action in legalActions(node[-1][0], node[-1][1]): # xét các nước đi hợp lệ từ node hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # tính toán trạng thái sau khi đi một nước
                if isFailed(newPosBox): # Nếu nước này dẫn box tới trạng thái fail (bị kẹt và không thể di chuyển được) thì bỏ qua
                    continue
                newCost = cost(node_action[1:])+1
                frontier.push(node + [(newPosPlayer, newPosBox)], newCost) # Cập nhật node với độ ưu tiên lớn hơn node hiện tại 1
                actions.push(node_action + [action[-1]], cost(node_action[1:])) # Cập nhật đường đi với prority là độ dài mới

    return temp # Sau khi tìm được lời giải thì trả về các nước đã đi


def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        # distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1])) # available heuristic
        distance += (abs(posPlayer[0] - sortposBox[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1])) # my heuristic

    return distance

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    # Các node (state từ bắt đầu đến hiện tại) được tổ chức thành hàng đợi ưu tiên
    # với độ ưu tiên là heuristic của trạng thái cuối cùng
    start = time.time()
    beginBox = PosOfBoxes(gameState) # vị trí ban đầu của các hộp
    beginPlayer = PosOfPlayer(gameState) # vị trí ban đầu của người chơi
    temp = [] # mảng ghi lại các nước đã đi
    start_state = (beginPlayer, beginBox) # node đầu tiên
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox)) # gọi hàm heuristic cho node đầu tiên và set nó làm độ ưu tiên
    exploredSet = set() # ghi lại những node đã có để không lặp lại
    actions = PriorityQueue() # ghi lại những nước đã đi ứng với từng node
    actions.push([0], heuristic(beginPlayer, start_state[1]))
    while len(frontier.Heap) > 0: # tiếp tục vòng lặp khi vẫn còn trạng thái chưa xét
        node = frontier.pop() # lấy ra node có độ ưu tiên thấp nhất trong hàng đợi
        node_action = actions.pop() # lấy ra action có độ ưu tiên thấp nhất trong hàng đợi
        if isEndState(node[-1][-1]): # nếu trạng thái cuối cùng của node là trạng thái kết thúc thì kết trúc trò chơi
            temp += node_action[1:] # trả về các nước đã đi
            break

        ### CONTINUE YOUR CODE FROM HERE
        if node[-1] not in exploredSet:  # Nếu node hiện tại chưa xét thì xét, nếu rồi thì xét node khác
            exploredSet.add(node[-1])  # add node hiện tại vào danh sách các node đã xét
            for action in legalActions(node[-1][0], node[-1][1]):  # xét các nước đi hợp lệ từ node hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)  # tính toán trạng thái sau khi đi một nước
                if isFailed(newPosBox):  # Nếu nước này dẫn box tới trạng thái fail (bị kẹt và không thể di chuyển được) thì bỏ qua
                    continue
                newDistance = heuristic(newPosPlayer, newPosBox) # tính toán heuristic của trạng thái tiếp theo
                frontier.push(node + [(newPosPlayer, newPosBox)],
                              newDistance)  # Cập nhật node với độ ưu tiên là heuristic
                actions.push(node_action + [action[-1]],
                             heuristic(newPosPlayer, newPosBox))  # Cập nhật đường đi

    end = time.time()

    return temp, end-start

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    print("layout=", layout)
    print("player_pos=", player_pos)
    # time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)

    run_time = 0
    
    if method == 'dfs':
        result, run_time = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result, run_time = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result, run_time = uniformCostSearch(gameState)
    elif method == 'astar':
        result, run_time = aStarSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    # time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, run_time))
    print('Step counted: ', len(result))
    print(result)
    return result
