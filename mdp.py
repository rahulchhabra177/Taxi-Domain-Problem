import numpy as np
import random




GRID_ROWS = 5
GRID_COLS = 5
STATE_R = (0,0)
STATE_G = (0,4)
STATE_Y = (4,0)
STATE_B = (4,3)
VALUES = np.zeros(shape = (GRID_ROWS, GRID_COLS))
GAMMA = 0.9 # discount factor
ALPHA = 0.1 # learning rate
NUM_EPISODES = 1000

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
    def __init__(self, taxi_loc, passenger_loc,passenger_sitting):
        
        self.taxi_loc = taxi_loc
        self.passenger_loc = passenger_loc
        self.passenger_sitting = passenger_sitting
        self.finished = False
        self.possible_states = []
        self.drop_loc = drop_loc
    def val(self):
        return (self.taxi_loc,self.passenger_loc,self.drop_loc,int(self.passenger_sitting))

    def initialize(self) :
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                for a in range(GRID_ROWS):
                    for b in range(GRID_COLS):
                        cur_state = State((i,j),(a,b),False)
                        self.possible_states.append(cur_state) 
        
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                self.possible_states.append(State((i,j),(i,j),True))       



    def __str__(self):
        grid = ["+---------+","|R:_|_:_:G|","|_:_|_:_:_|","|_:_:_:_:_|","|_|_:_|_:_|","|Y|_:_|B:_|","+---------+"]
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
                return State(taxi_loc,passenger_loc,True) 
            else:    
                return self
        else:
            if taxi_loc == drop_loc and self.passenger_sitting:
                # self.passenger_sitting = False
                # self.finished = True
                stt = State(taxi_loc, taxi_loc, False)
                stt.finished = True 
                return stt
            elif self.passenger_sitting:
                return State(taxi_loc, taxi_loc, False)    
            else:
                return self

        if self.passenger_sitting:
                return State(taxi_loc, taxi_loc, True)  

        return State(taxi_loc, passenger_loc, self.passenger_sitting)                                           



class Agent:
    def __init__(self, state):
        self.state = state
        self.reward = 0

    def get_action(self):
        return np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.1,0.1,0.3,0.3,0.1,0.1])

    def update(self, action_intended):
        action = self.state.choose_action(action_intended)
        self.reward += self.state.get_reward(action)
        self.state = self.state.next_state(action)

    def play(self):
        action_intended = self.get_action()
        
        self.update(action_intended)
        return self.state, self.reward, action_intended

    def learn_optimal_values(self):
        converged = False
        num_iter = 0
        self.old_values = [np.zeros(shape = (GRID_ROWS, GRID_COLS)) for i in range(2)]
        self.new_values = [np.zeros(shape = (GRID_ROWS, GRID_COLS)) for i in range(2)]
        
        while not converged:
            num_iter = num_iter + 1
            for p_pick in [True, False]:
                for i in range(GRID_ROWS):
                    for j in range(GRID_COLS): 
                        if p_pick:                      
                            cur_state = State((i,j),(i,j),p_pick)
                        else:
                            cur_state = State((i,j),state.passenger_loc,p_pick)

                        max_value = -100000
                        print("cur_state:", cur_state.val())
                        # print(cur_state)
                        for action in ["N","S","E","W","pickup","putdown"]:
                            value = 0
                            if action == "pickup" :
                                reward_val = cur_state.get_reward(action)
                                next_state = cur_state.next_state(action)
                                if next_state.passenger_sitting and not p_pick:
                                    value = reward_val + GAMMA * self.old_values[1][next_state.taxi_loc[0],next_state.taxi_loc[1]]
                                else:
                                    value = reward_val + GAMMA * self.old_values[0][next_state.taxi_loc[0]][next_state.taxi_loc[1]]
                                print(action,reward_val,value)
                                # print(next_state)                          
                            elif action == "putdown":
                                reward_val = cur_state.get_reward(action)
                                next_state = cur_state.next_state(action)
                                if cur_state.taxi_loc == cur_state.drop_loc and p_pick:       
                                    value = reward_val + GAMMA * self.old_values[1][next_state.taxi_loc[0]][next_state.taxi_loc[1]]
                                    print(action,reward_val,value)
                                else:
                                    value = reward_val + GAMMA * self.old_values[0][next_state.taxi_loc[0]][next_state.taxi_loc[1]]
                                print(action,reward_val,value)
                                # print(next_state)
                            else:
                                for pos_action in ["N","S","E","W"]:
                                    reward_val = cur_state.get_reward(pos_action)
                                    next_state = cur_state.next_state(pos_action)
                                    # if next_state == cur_state:
                                    #     continue
                                    print(action,pos_action,reward_val,value)    
                                    if pos_action == action:
                                        value += 0.85*(reward_val + GAMMA * self.old_values[int(p_pick)][next_state.taxi_loc[0]][next_state.taxi_loc[1]])
                                    else:
                                        value+= 0.15 * (reward_val + GAMMA * self.old_values[int(p_pick)][next_state.taxi_loc[0]][next_state.taxi_loc[1]])        
                                    # print(next_state)
                            if value > max_value:
                                max_value = value
                        
                        self.new_values[int(p_pick)][i][j] = max_value
                               
                self.old_values[0] = self.new_values[0]
                self.old_values[1] = self.new_values[1]
            if num_iter > 6:
                break
    def extract_policy(self):
        self.policy = np.array([[["A"for k in range(2)] for i in range(GRID_ROWS)] for j in range(GRID_COLS)])
        for k in [True, False]:
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    cur_state = State((i,j),state.passenger_loc,k)
                    max_value = -100000
                    for action in ["N","S","E","W","pickup","putdown"]:
                        value = 0
                        if action == "pickup" or action == "putdown":
                            reward_val = cur_state.get_reward(action)
                            next_state = cur_state.next_state(action)
                            value = reward_val + GAMMA * self.new_values[int(next_state.passenger_sitting)][next_state.taxi_loc[0]][next_state.taxi_loc[1]]
                        else:
                            for pos_action in ["N","S","E","W"]:
                                reward_val = cur_state.get_reward(pos_action)
                                next_state = cur_state.next_state(pos_action)
                                # if next_state == cur_state:
                                #     continue
                                if pos_action == action:
                                    value += 0.85*(reward_val + GAMMA * self.new_values[int(next_state.passenger_sitting)][next_state.taxi_loc[0]][next_state.taxi_loc[1]])
                                else:
                                    value+= 0.15 * (reward_val + GAMMA * self.new_values[int(next_state.passenger_sitting)][next_state.taxi_loc[0]][next_state.taxi_loc[1]])        
                        if value > max_value:
                            max_value = value
                            self.policy[i][j][int(k)] = action
        return self.policy

    def index_to_action(self, index):
        if  index == 0 :
            return "N"
        elif index == 1:
            return "S"
        elif index == 2:
            return "E"
        elif index == 3:
            return "W"
        elif index == 4:
            return "pickup"
        elif index == 5:
            return "putdown"
        else:
            return "N"            
    def q_learning(self):
        """N,S,E,W,pickup,Drop"""
        for episode_num in range(NUM_EPISODES):
            self.q_table = np.zeros(shape = (GRID_ROWS, GRID_COLS, 6))
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    random_val = random.uniform(0,1)
                    action = ""
                    action_num = 0
                    if random_val < 0.1:
                        action_num = np.argmax(self.q_table[i,j,:])
                    else:
                        action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
                    
                    action = self.index_to_action(action_num)
                    reward = self.state.get_reward(action)
                    next_state = self.state.next_state(action)
                    self.q_table[i][j][action_num]  =   self.q_table[i][j][action_num] + ALPHA * (reward + GAMMA * np.max(self.q_table[next_state.taxi_loc[0]][next_state.taxi_loc[1]]) - self.q_table[i][j][action_num])  
                    self.state = next_state




drop_loc = STATE_Y                        
state = State((4,4), STATE_G, False)
agent = Agent(state)
agent.learn_optimal_values()
print(agent.new_values[0])
print(agent.new_values[1])

agent.extract_policy()
print(agent.policy[:,:,0])
print(agent.policy[:,:,1])

# for i in range(100):
#     print(state)
#     state = state.next_state(agent.policy[state.taxi_loc[0]][state.taxi_loc[1]])
#     if state.taxi_loc == state.drop_loc:
#         print("You reached your destination")
#         break
