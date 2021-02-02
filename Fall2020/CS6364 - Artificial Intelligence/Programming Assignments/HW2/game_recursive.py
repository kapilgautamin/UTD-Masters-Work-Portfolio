import copy
from collections import deque
# import sys
# sys.setrecursionlimit(8000)

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
    def __init__(self, name, start: Position, goal: Position,loc, utility=None, father=None):
        self.name = str(name)
        self.pos = start
        self.goal = goal
        self.father = father
        self.utility = utility
        self.loc = loc

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
        
        return zip(actions, actions_name)

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

class Game:
    def __init__(self, players: [Player]):
        self.players = players
        self.explored_node = []
        # self.show_player_positions()

    def check_repeated_state(self, game_state, new_node):
        # print("repeated sec game state")
        # self.show_game_state(game_state)
        # print("repeated sec explored nodes")   
        # self.show_game_state(self.explored_node)
        count_pos = 0
        for node in game_state:
            count = 0
            for px, py in zip(new_node, node):
                # print(px,py)
                if px == py:
                    count += 1
            if count == 4:
                # print("REPEATED node at pos",count_pos)
                # for player in node:
                #     print(player, sep=' ')
                # self.show_game_state(game_state)
                return True
            count_pos += 1

        count_pos = 0    
        for node in self.explored_node:
            count = 0
            for px, py in zip(new_node, node):
                # print(px,py)
                if px == py:
                    count += 1
            if count == 4:
                # print("REPEATED node at pos",count_pos)
                # for player in node:
                #     print(player, sep=' ')
                # self.show_game_state(game_state)
                return True
            count_pos += 1
        return False

    def play_game(self, game_state, player_loc=0,depth=0):
        if depth >= 15:
            # print("Maximum depth of 15 reached")
            return (0, 0, 0, 0)
        player = self.players[player_loc]
        player_actions = player.actions(player.pos)
        # need a copy of game state before we do any changes
        update_game_state = copy.deepcopy(game_state)
        
        for action, action_name in player_actions:
            # print(depth,"Depth,",player,"playing, move", action_name)
            old_node = copy.deepcopy([player.pos for player in self.players])
            if self.other_player_already_present(player, action) == False:

                # store action to revert it if repeated state
                revert_action = Position(player.pos.x, player.pos.y)
                # below move updates node reference as well
                player.make_move(action)
                # self.show_player_positions()
                node = [player.pos for player in self.players]
                # player.update_father(game_state, old_node)
                if self.check_repeated_state(update_game_state, node):
                    print("Depth={}|Current player={}| Father node={}| Action={}| Current node={}| REPEATED".format(depth, player.name,
                                        self.get_compact_state(old_node),action_name,self.get_compact_state(node)))
                    # print("Current player={}, Father node={}, Action={}, Current node={}, REPEATED".format(player.name,
                    #                                                                                        self.get_compact_state(player.father), action_name, self.get_compact_state(node)))
                    player.make_move(revert_action)
                    #Successors of repeated game nodes should not be considered!
                    return (None,None,None,None)
                else:
                    if player.goal_test():
                        print("Depth={}|Current player={}| Father node={}| Action={}| Current node={}| WINS={}".format(depth, player.name,
                                    self.get_compact_state(old_node),action_name,self.get_compact_state(node),player.name))
                        # print("Current player={}, Father node={}, Action={}, Current node={}, WINS={}".format(player.name,
                        #                                                                                       self.get_compact_state(player.father), action_name, self.get_compact_state(node), player.name))
                        print(player.terminal_value())
                        self.explored_node.append(node)
                        self.explored_node = copy.deepcopy(self.explored_node)
                        for father,play in zip(old_node,self.players):
                            play.pos = father
                    else:
                        print("Depth={}| Current player={}| Father node={}| Action={}| Current node={}".format(depth, player.name,
                                        self.get_compact_state(old_node),action_name,self.get_compact_state(node)))
                        # print("Current player={}, Father node={}, Action={}, Current node={}".format(player.name,
                        #                                                                              self.get_compact_state(player.father), action_name, self.get_compact_state(node)))
                        # print("print node to be added in explored list", self.get_compact_state(node))
                        self.explored_node.append(node)
                        self.explored_node = copy.deepcopy(self.explored_node)
                        update_game_state.append(node)
                        self.play_game(update_game_state,(player_loc+1)%4, depth+1)

                        # backtracking
                        update_game_state.pop()
                        # print("revert back",depth,"to player",player)
                        # print("old",self.get_compact_state(old_node))
                        for father,play in zip(old_node,self.players):
                            play.pos = father


    def other_player_already_present(self, curr_player, action):
        for player in self.players:
            # print(player.name, curr_player.name, player.pos, action)
            if player.name != curr_player.name and player.pos == action:
                # print("Revert, ",player," player already present at position",action, "for player", curr_player)
                return True
        return False

    def get_compact_state(self, values):
        compact_state = []
        for player in values:
            if type(player) == type(self.players[0]):
                compact_state.append((player.pos.x, player.pos.y))
            else:
                compact_state.append((player.x, player.y))
        return compact_state

    def show_player_positions(self):
        positions = self.get_compact_state(self.players)
        print("Current player positions", positions)

    def show_game_state(self, game_state):
        print("Proceed", len(game_state))
        for state in range(len(game_state)):
            compact_state = self.get_compact_state(game_state[state])
            print(state, compact_state)

if __name__ == '__main__':
    P1 = Player('P1', start=Position(1, 1), goal=Position(4, 4),loc=0)
    P2 = Player('P2', start=Position(4, 1), goal=Position(1, 4),loc=1)
    P3 = Player('P3', start=Position(4, 4), goal=Position(1, 1),loc=2)
    P4 = Player('P4', start=Position(1, 4), goal=Position(4, 1),loc=3)

    # Position of players make them go one by one as given in the following order
    game_players = [P1, P2, P3, P4]
    # four_player_minimax.show_player_positions()

    four_player_minimax = Game(game_players)
    initial_game_state = []
    node = [player.pos for player in game_players]
    initial_game_state.append(node)
    four_player_minimax.play_game(initial_game_state)
