import copy
from collections import deque

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
    def __init__(self, name, start: Position, goal: Position,loc, utility=None):
        self.name = str(name)
        self.pos = start
        self.goal = goal
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

    def __str__(self):
        return 'name:' + self.name + ' pos:' + str(self.pos)

class Iterative_Game:
    def __init__(self, gamers):
        self.players = gamers
        self.initial = (None, None, "", self.get_compact_state(self.players), self.players[0])

    def play_game(self):
        game_queue = deque()
        explored_nodes = set()
        # push root node
        game_queue.append(self.initial)        
        explored_nodes.add(self.initial[3])

        while game_queue:
            old_game_node = game_queue.popleft()
            # print("length of game queue", len(game_queue))
            curr_player = old_game_node[0]
            action = old_game_node[1]
            action_name = old_game_node[2]
            state = old_game_node[3]
            next_player = old_game_node[4]
            if curr_player == None:
                #game starting
                player_actions = next_player.actions(next_player.pos)
                #enque successors
                for act, act_name in player_actions:
                    queue_curr_player = self.players[0]
                    queue_next_player = self.players[1]
                    game_node = self.create_game_node(queue_curr_player, act, act_name, self.players, queue_next_player)
                    game_queue.append(game_node)
            else:
                # print("Current player:{}, Next player:{}".format(curr_player.name, next_player.name))

                # print("{} playing move {}".format(curr_player.name,action_name))
                revert_action = Position(curr_player.pos.x, curr_player.pos.y)
                for player,positions in zip(self.players,state):
                    player.pos.x = positions[0]
                    player.pos.y = positions[1]
    
                for player in self.players:
                    if curr_player.name == player.name:
                        player.make_move(action)
                        curr_player = player
                        break
                new_state = self.get_compact_state(self.players)

                if new_state in explored_nodes:
                    print("[Current player={} | Father node={} | Action={} | Current game node={} | REPEATED]".format(curr_player.name,
                            state, action_name, new_state))
                    for player in self.players:
                        if curr_player.name == player.name:
                            player.make_move(revert_action)
                else:
                    explored_nodes.add(new_state)
                    if curr_player.goal_test():
                        print("[Current player={} | Father node={} | Action={} | Current game node={} | WINS=[PLAYER {}]]".format(curr_player.name,
                                state, action_name, new_state, curr_player.name))
                        # print(curr_player.terminal_value())
                    else:
                        print("[Current player={} | Father node={} | Action={} | Current node={}]".format(curr_player.name,
                                state, action_name, new_state))
                        player_actions = next_player.actions(next_player.pos)
                        #enque successors
                        for act, act_name in player_actions:
                            if self.other_player_already_present(next_player, act) == False:
                                queue_curr_player = self.players[(next_player.loc) % 4]
                                queue_next_player = self.players[(next_player.loc+1) % 4]
                                game_node = self.create_game_node(queue_curr_player, act, act_name, self.players, queue_next_player)
                                game_queue.append(game_node)           
            # self.show_game_queue(game_queue)

    def other_player_already_present(self, curr_player, action):
        for player in self.players:
            # print(player.name, curr_player.name, player.pos, action)
            if player.name != curr_player.name and player.pos == action:
                # print("Revert, ",player," player already present at position",action, "for player", curr_player)
                return True
        return False
    
    def create_game_node(self, curr_player,action,action_name, player_stats, next_player):      
        state = self.get_compact_state(player_stats)
        curr_player = copy.deepcopy(curr_player)
        next_player = copy.deepcopy(next_player)
        return (curr_player, action, action_name, state, next_player)

    def get_compact_state(self, state):
        state = copy.deepcopy(state)
        compact_state = []
        for node in state:
            #extract the game state from players class
            if isinstance(node, Player):
                compact_state.append((node.pos.x, node.pos.y))
            #extract the game state from positions
            else:
                compact_state.append((node.x, node.y))
        return tuple(compact_state)

    def show_player_positions(self):
        positions = self.get_compact_state(self.players)
        print("Current player positions", positions)

    def show_game_queue(self, gameplay):
        print("printing game queue")
        for game_node in gameplay:
            curr_player = game_node[0].name
            act_name = game_node[2]
            state = game_node[3]
            next_player = game_node[4].name
            print(curr_player,act_name,state,next_player)

if __name__ == '__main__':
    P1 = Player('P1', start=Position(1, 1), goal=Position(4, 4), loc=0)
    P2 = Player('P2', start=Position(4, 1), goal=Position(1, 4), loc=1)
    P3 = Player('P3', start=Position(4, 4), goal=Position(1, 1), loc=2)
    P4 = Player('P4', start=Position(1, 4), goal=Position(4, 1), loc=3)

    game_players = [P1, P2, P3, P4]
    # four_player_minimax.show_player_positions()

    four_player_minimax = Iterative_Game(game_players)
    four_player_minimax.play_game()
