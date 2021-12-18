import numpy as np
import random
import sys
STATE_R = (0,0)
STATE_G = (0,4)
STATE_Y = (4,0)
STATE_B = (4,3)
GAMMA = 0.99     # discount factor
ALPHA = 0.25           # learning rate
NUM_EPISODES = 2000
EPISODE_SIZE = 500
drop_loc = STATE_G
print("x,y")
# print(sys.argv[0])
# Action Space = [North, South, East, West, Pickup, Putdown]
# State space will have three things: current position of taxi, passenget location, drop location

# Model the taxi domain problem with 5*5 grid world using markov decision process where the taxi can move in four directions and pickup and drop passengers at any location in the grid world .
"""
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
"""
class State:
    def __init__(self, taxi_loc, passenger_loc,passenger_sitting,is_10):
        
        self.taxi_loc = taxi_loc
        self.passenger_loc = passenger_loc
        self.passenger_sitting = passenger_sitting
        self.finished = False
        self.possible_states = []
        self.drop_loc = drop_loc
        self.is_10_by_10 = is_10
    def val(self):
        return (self.taxi_loc,self.passenger_loc,int(self.passenger_sitting))

    def initialize(self) :
        grid_rows = 5
        grid_cols = 5

        if self.is_10_by_10:
            grid_rows = 10
            grid_cols = 10

        for i in range(grid_rows):
            for j in range(grid_cols):
                for a in range(grid_rows):
                    for b in range(grid_cols):
                        cur_state = State((i,j),(a,b),False,self.is_10_by_10)
                        self.possible_states.append(cur_state) 
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                self.possible_states.append(State((i,j),(i,j),True,self.is_10_by_10))       



    def __str__(self):
        grid = [
            "+---------+",
            "|R:_|_:_:G|",
            "|_:_|_:_:_|",
            "|_:_:_:_:_|",
            "|_|_:_|_:_|",
            "|Y|_:_|B:_|",
            "+---------+"
            ]
        if self.is_10_by_10:
           grid = [
                "+--------------------+",
                "|R:_:_|_:_:G:_:_|C:_|",
                "|_:_:_|_:_:_:_:_|_:_|",
                "|_:_:_|_:_:_|_:_|_:_|",
                "|_:_:_|W:_:_|_:_|_:_|",
                "|_:_:_:_:_:_|M:_:_:_|",
                "|_:_:_:_:_:_|_:_:_:_|",
                "|_|_:_:_|_:_:_:_|_:_|",
                "|_|_:_:_|_:_:_:_|_:_|",
                "|_|_:_:_|_:_:_:_|_:_|",
                "|Y|_:_:_|B:_:_:_|_:P|",
                "+--------------------+"
                ]     
        taxi_loc = self.taxi_loc
        taxi_x = taxi_loc[0]
        taxi_y = taxi_loc[1]
        CRED = '\033[33m'
        CEND = '\033[0m'
        grid[taxi_x+1] = grid[taxi_x+1][:taxi_y*2+1] + CRED + "T" + CEND + grid[taxi_x+1][taxi_y*2+2:]
        result = ""
        style, fg, bg = 0,30,47
        format = ';'.join([str(style), str(fg), str(bg)])
        for grids in grid:
            result = result + grids + "\n"
        return result     

    def __eq__(self, other):
        return self.taxi_loc == other.taxi_loc and self.passenger_loc == other.passenger_loc and self.passenger_sitting == other.passenger_sitting
  
    def __hash__(self):
        return hash((self.taxi_loc,self.passenger_loc,int(self.passenger_sitting)))
  
    def get_reward(self, action):
        if action == "pickup":
            if self.taxi_loc == self.passenger_loc:
                return -1
            else:
                return -10

        elif action == "putdown":
            if self.taxi_loc == self.drop_loc and self.passenger_sitting:
                return 20
            elif self.passenger_sitting or self.taxi_loc == self.passenger_loc:
                return -1    
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
        if self.is_10_by_10:
            return self.next_state_10(action)
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

        elif action == "pickup":
            if taxi_loc == passenger_loc and not self.passenger_sitting:
                return State(taxi_loc,passenger_loc,True, self.is_10_by_10) 
            else:    
                return self
        else:
            if taxi_loc == drop_loc and self.passenger_sitting:
                # self.passenger_sitting = False
                # self.finished = True
                stt = State(taxi_loc, taxi_loc, False, self.is_10_by_10)
                stt.finished = True 
                return stt
            elif self.passenger_sitting:
                return State(taxi_loc, taxi_loc, False, self.is_10_by_10)    
            else:
                return self

        if self.passenger_sitting:
                return State(taxi_loc, taxi_loc, True, self.is_10_by_10)  

        return State(taxi_loc, passenger_loc, self.passenger_sitting, self.is_10_by_10)                                           

    def next_state_10(self,action):
        taxi_loc = self.taxi_loc
        passenger_loc = self.passenger_loc
        drop_loc = self.drop_loc
        if action == "N":
            taxi_loc = (max(0,taxi_loc[0]-1),taxi_loc[1])
        elif action == "S":
            taxi_loc = (min(9,taxi_loc[0]+1),taxi_loc[1])
        elif action == "E":
            if taxi_loc[1]==0 and taxi_loc[0] > 5:
                return self
            elif taxi_loc[1] == 2 and taxi_loc[0] < 4:
                return self    
            elif taxi_loc[1] == 3 and taxi_loc[0] > 5:
                return self    
            elif taxi_loc[1] == 5 and taxi_loc[0] > 1 and taxi_loc[0] < 6:
                return self    
            elif taxi_loc[1] == 7 and (taxi_loc[0] <4 or taxi_loc[0] > 5):
                return self        
            else:
                taxi_loc = (taxi_loc[0],min(taxi_loc[1]+1, 9 ))

        elif action == "W":
            if taxi_loc[1]==1 and taxi_loc[0] > 5:
                return self
            elif taxi_loc[1] == 3 and taxi_loc[0] < 4:
                return self    
            elif taxi_loc[1] == 4 and taxi_loc[0] > 5:
                return self    
            elif taxi_loc[1] == 6 and taxi_loc[0] > 1 and taxi_loc[0] < 6:
                return self    
            elif taxi_loc[1] == 8 and (taxi_loc[0] <4 or taxi_loc[0] > 5):
                return self        
            else:
                taxi_loc = (taxi_loc[0],max(taxi_loc[1] - 1, 0 ))

        
        elif action == "pickup":
            if taxi_loc == passenger_loc and not self.passenger_sitting:
                return State(taxi_loc,passenger_loc,True, self.is_10_by_10) 
            else:    
                return self
        else:
            if taxi_loc == drop_loc and self.passenger_sitting:
            
                stt = State(taxi_loc, taxi_loc, False, self.is_10_by_10)
                stt.finished = True 
                return stt
            elif self.passenger_sitting:
                return State(taxi_loc, taxi_loc, False, self.is_10_by_10)    
            else:
                return self

        if self.passenger_sitting:
                return State(taxi_loc, taxi_loc, True, self.is_10_by_10)  

        return State(taxi_loc, passenger_loc, self.passenger_sitting, self.is_10_by_10)                                           





