#Position class to store the x,y coordinates for the players in the game
class Position:
    __slots__ = ["x", "y"]

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return '{x:' + str(self.x) + ' y:' + str(self.y) + '}'

class Player:
    def __init__(self, name, start: Position, goal: Position, utility=None, father=None):
        self.name = str(name)
        self.pos = start
        self.goal = goal
        self.father = father
        self.utility = utility

    def goal_test(self):
        return self.goal == self.pos

    def actions(self, coor):
        actions = []
        actions_name = []
        up = Position(coor.x, coor.y-1)
        down = Position(coor.x, coor.y+1)
        right = Position(coor.x+1, coor.y)
        left = Position(coor.x-1, coor.y)
        # clockwise add up,right,down,left
        if self.move_legal(up):
            actions.append(up)
            actions_name.append('UP')
        if self.move_legal(right):
            actions.append(right)
            actions_name.append('RIGHT')
        if self.move_legal(down):
            actions.append(down)
            actions_name.append('DOWN')
        if self.move_legal(left):
            actions.append(left)
            actions_name.append('LEFT')
        
        return zip(actions,actions_name)

    def move_legal(self, loc):
        if(loc.x <= 4 and loc.x >= 1 and loc.y >= 1 and loc.y <= 4):
            return True
        return False

    def make_move(self, act):
        self.pos.x = act.x
        self.pos.y = act.y

    def terminal_value(self):
        # print("winning player",self.name)
        if self.name == 'P1':
            return (200, 10, 30, 10)
        elif self.name == 'P2':
            return (100, 300, 150, 200)
        elif self.name == 'P3':
            return (150, 200, 400, 300)
        elif self.name == 'P4':
            return (220, 330, 440, 500)
        else:
            print("Something's fishy")
            return (0, 0, 0, 0)

    def update_father(self, gameplay, father_game_node):
        self.gameplay = gameplay
        self.father = father_game_node

    def __str__(self):
        return 'name:' + self.name + ' pos:' + str(self.pos)

class Test:
   def __init__(self, players):
      self.recursion_limit = 6
      self.players = players
      self.recursion = 0

   def get_compact_state(self, values):
        compact_state = []
        for player in values:
            if type(player) == type(self.players[0]):
                compact_state.append((player.pos.x, player.pos.y))
            else:
                compact_state.append((player.x, player.y))
        return compact_state

   def print_it(self,loc,depth):
      player = self.players[loc]
      player_actions = ['L','R']
      
      for action in player_actions:
         print(player," playing ",action," with recursion", self.recursion, " and depth ", depth)
         if self.recursion > self.recursion_limit:
            print(player,"Player won, reached recursion limit")
            return self.recursion
         self.recursion += 1
         utility = self.print_it((loc+1)%4,depth+1)
         print("utility value", utility)


P1 = Player('P1', start=Position(1, 1), goal=Position(4, 4))
P2 = Player('P2', start=Position(4, 1), goal=Position(1, 4))
P3 = Player('P3', start=Position(4, 4), goal=Position(1, 1))
P4 = Player('P4', start=Position(1, 4), goal=Position(4, 1))
players = ['P1','P2','P3','P4']
t = Test(players)
t.print_it(0,0)