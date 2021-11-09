import numpy as np

GRID_ROWS = 5
GRID_COLS = 5
STATE_R = (0,0)
STATE_G = (0,4)
STATE_Y = (4,0)
STATE_B = (4,3)

# Action Space = [North, South, East, West, Pickup, Putdown]
# State space will have three things: current position of taxi, passenget location, drop location

# Model the taxi domain problem with 5*5 grid world using markov decision process where the taxi can move in four directions and pickup and drop passengers at any location in the grid world .


class State:
    def __init__(self, taxi_loc, passenger_loc, drop_loc):
        self.taxi_loc = taxi_loc
        self.passenger_loc = passenger_loc
        self.drop_loc = drop_loc

    def __str__(self):
        grid = ["-------------","|R._|_._._.G|","|_._|_._._._|","|_._._._._._|","|_|_._|_._._|","|Y|_._|_.B._|","-------------"]
        taxi_loc = self.taxi_loc
        taxi_x = taxi_loc[0]
        taxi_y = taxi_loc[1]
        grid[taxi_y + 1][taxi_x * 2 + 1] = "T"
        result = ""
        for grids in grid:
            result = result + grids + "\n"
        return result     


    def __eq__(self, other):
        return self.taxi_loc == other.taxi_loc and self.passenger_loc == other.passenger_loc and self.drop_loc == other.drop_loc

    def get_reward(self, action):
        if action == "pickup":
            if self.taxi_loc == self.passenger_loc:
                return -10
            else:
                return -1
        elif action == "putdown":
            if self.taxi_loc == self.drop_loc:
                return 20
            else:
                return -10
        else:
            return -1


    def choose_action(self,action):
        if action == "N":
            return np.random.choice(["N","S","E","W"],p=[0.85,0.05,0.05,0.05])
        elif action == "S":
            return np.random.choice(["N","S","E","W"],p=[0.05,0.85,0.05,0.05])
        elif action == "E":
            return np.random.choice(["N","S","E","W"],p=[0.05,0.05,0.85,0.05])
        elif action == "W": 
            return np.random.choice(["N","S","E","W"],p=[0.05,0.05,0.05,0.85])
        else:
            return action 

    def next_state(self,action):
        taxi_loc = self.taxi_loc
        passenger_loc = self.passenger_loc
        drop_loc = self.drop_loc
        if action == "N":
            taxi_loc = (max(0,taxi_loc[0]-1),taxi_loc[1])
        elif action == "S":
            taxi_loc = (min(4,taxi_loc[0]+1),taxi_loc[1])
        elif action == "E":
            if taxi_loc[1]==0 and taxi_loc[0] > 2:
                return self
            elif taxi_loc[1] == 1 and taxi_loc[0] < 2:
                return self    
            elif taxi_loc[1] == 2 and taxi_loc[0] > 2:
                return self    
            else:
                taxi_loc = (taxi_loc[0],min(taxi_loc[1]+1, 4 ))

        elif action == "W":
            if taxi_loc[1]==1 and taxi_loc[0] > 2:
                return self
            elif taxi_loc[1] == 2 and taxi_loc[0] < 2:
                return self    
            elif taxi_loc[1] == 3 and taxi_loc[0] > 2:
                return self    
            else:
                taxi_loc = (taxi_loc[0],max(taxi_loc[1] - 1, 0 ))

        else:
            return self

        return State(taxi_loc, passenger_loc, drop_loc)                                           





class Agent:
    def __init__(self, state):
        self.state = state
        self.reward = 0

    def get_action(self):
        return np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.1,0.1,0.1,0.1,0.5,0.5])

    def update(self, action_intended):
        action = self.state.choose_action(action_intended)
        self.reward += self.state.get_reward(action)
        self.state = self.state.next_state(action)

    def play(self):
        action_intended = self.get_action()
        self.update(action_intended)
        return self.state, self.reward, action_intended