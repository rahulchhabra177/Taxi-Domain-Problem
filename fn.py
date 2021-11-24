import numpy as np
import random




GRID_ROWS = 5
GRID_COLS = 5
STATE_R = (0,0)
STATE_G = (0,4)
STATE_Y = (4,0)
STATE_B = (4,3)
VALUES = np.zeros(shape = (GRID_ROWS, GRID_COLS))
GAMMA = 0.9       # discount factor
ALPHA = 0.1       # learning rate
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
        return (self.taxi_loc,self.passenger_loc,int(self.passenger_sitting))

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
        return np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.2,0.2,0.2,0.2,0.1,0.1])

    def update(self, action_intended):
        action = self.state.choose_action(action_intended)
        print(action_intended, action)
        self.reward += self.state.get_reward(action)
        self.state = self.state.next_state(action)

    def play(self):
        action_intended = self.get_action()
        print(action_intended)
        self.update(action_intended)
        return self.state, self.reward, action_intended

    def learn_optimal_values(self):
        converged = False
        num_iter = 0
        self.old_values = {}
        for state in self.state.possible_states:
            self.old_values[state.val()] = 0
        self.new_values = {}
        
        while not converged:
            num_iter = num_iter + 1
            max_norm = -10000
            for state in self.state.possible_states:
                cur_state = State(state.taxi_loc, state.passenger_loc, state.passenger_sitting)
                max_value = -100000
                for action in ["N","S","E","W","pickup","putdown"]:
                    value = 0
                    if action == "pickup" :
                        reward_val = cur_state.get_reward(action)
                        next_state = cur_state.next_state(action)
                        value = reward_val + GAMMA * self.old_values[next_state.val()]
                    elif action == "putdown":
                        reward_val = cur_state.get_reward(action)
                        next_state = cur_state.next_state(action)
                        if cur_state.taxi_loc == cur_state.drop_loc and cur_state.passenger_sitting:
                            value = reward_val
                        else:    
                            value = reward_val + GAMMA * self.old_values[next_state.val()] 
                    else:
                        for pos_action in ["N","S","E","W"]:
                            reward_val = cur_state.get_reward(pos_action)
                            next_state = cur_state.next_state(pos_action)
                            # if next_state == cur_state:
                            #     continue
                            if pos_action == action:
                                value += 0.85*(reward_val + GAMMA * self.old_values[next_state.val()])
                            else:
                                value += 0.05*(reward_val + GAMMA * self.old_values[next_state.val()])
                                
                    if value > max_value:
                        max_value = value
                # print(max_value)   
                # if cur_state.taxi_loc == (0,4):
                #     print(cur_state.val())
                #     print(max_value)
                #     print(self.old_values[cur_state.val()])     
                self.new_values[cur_state.val()] = max_value
                # print(cur_state.val())
                max_norm = max(max_norm, abs(self.old_values[cur_state.val()] - max_value))
            
                # print(abs(self.old_values[cur_state.val()] - max_value),self.old_values[cur_state.val()], max_value)
                # print(cur_state.val(), max_value)               
            self.old_values = self.new_values.copy()
            print(num_iter,":", max_norm)

            if max_norm < 0.001:
                converged = True
            
            if num_iter > 500 or converged:
                break




    def generate_random_policy(self):
        policy_result = {}
        for state in self.state.possible_states:
            policy_result[state.val()] = np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.2,0.2,0.2,0.2,0.1,0.1])
        return policy_result
    
    # A function to evaluate given policy by iterating through all possible states
    # Return: a dictionary of state_value    
    def evaluate_policy_iterative(self, policy, values):
        state_value = {}
        for state in self.state.possible_states:
            cur_state = State(state.taxi_loc, state.passenger_loc, state.passenger_sitting)
            max_value = -100000
            action = policy[cur_state.val()]
            value = 0
            if action == "pickup" :
                reward_val = cur_state.get_reward(action)
                next_state = cur_state.next_state(action)
                value = reward_val + GAMMA * values[next_state.val()]
            elif action == "putdown":
                reward_val = cur_state.get_reward(action)
                next_state = cur_state.next_state(action)
                if cur_state.taxi_loc == cur_state.drop_loc and cur_state.passenger_sitting:
                    value = reward_val
                else:    
                    value = reward_val + GAMMA * values[next_state.val()]
            else:
                for pos_action in ["N","S","E","W"]:
                    reward_val = cur_state.get_reward(pos_action)
                    next_state = cur_state.next_state(pos_action)
                    # if next_state == cur_state:
                    #     continue
                    if pos_action == action:
                        value += 0.85*(reward_val + GAMMA * values[next_state.val()])
                    else:
                        value += 0.05*(reward_val + GAMMA * values[next_state.val()])
                        
            if value > max_value:
                max_value = value
                    
            state_value[cur_state.val()] = max_value
        return state_value            
    # a function to evaluate a given policy by using numpy linear algebra to solve the linear equation
    def evaluate_policy_linear_equation(self, policy):
        A = np.zeros((len(self.state.possible_states),len(self.state.possible_states)))
        b = np.zeros(len(self.state.possible_states))
    
    def policy_iteration(self): 
        policy = self.generate_random_policy()
        converged = False
        values = self.evaluate_policy_iterative(policy)
        num_iter = 0
        prev_policy = policy.copy()
        while not converged:
            prev_policy = policy.copy()
            policy = self.extract_policy(values)
            values = self.evaluate_policy_iterative(policy)
            print(num_iter)
            num_iter = num_iter + 1
            if num_iter > 200 or policy == prev_policy:
                converged = True
        return policy,values    

    def extract_policy(self, values):
        self.policy = {}
        for cur_state in self.state.possible_states:
            max_value = -100000
            next_state = None
            # print(cur_state.val())
            self.policy[cur_state.val()] = "U"
            for action in ["N","S","E","W","pickup","putdown"]:
                value = 0
                
                if action == "pickup" or action == "putdown":
                    reward_val = cur_state.get_reward(action)
                    next_state = cur_state.next_state(action)
                    # print("\n\n\n\nHrllo\n\n\n", cur_state.val(), next_state.val())
                    if action == "putdown" and cur_state.passenger_sitting and cur_state.taxi_loc == cur_state.drop_loc:
                        
                        value = reward_val 
                    else:    
                        value = reward_val + GAMMA * values[next_state.val()]
                    
                else:
                    for pos_action in ["N","S","E","W"]:
                        reward_val = cur_state.get_reward(pos_action)
                        next_state = cur_state.next_state(pos_action)
                        
                        if pos_action == action:
                            value += 0.85*(reward_val + GAMMA * values[next_state.val()])
                        else:
                            value+= 0.05 * (reward_val + GAMMA * values[next_state.val()])        
                print(cur_state.val(), action, value)
                if value >= max_value:                 
                    max_value = value
                    self.policy[cur_state.val()] = action

        return self.policy  

    def execute_policy(self, policy):
        cur_state = State(self.state.taxi_loc, self.state.passenger_loc, self.state.passenger_sitting)
        print(cur_state)
        total_reward = 0
        while not cur_state.finished:
            action = policy[cur_state.val()]
            print("Action Intended:",action)
            action = self.state.choose_action(action)
            print("Action Succeeded:", action)
            cur_reward = cur_state.get_reward(action)
            print("Reward:", cur_reward)
            total_reward += cur_reward
            print("Total Reward:", total_reward)
            cur_state = cur_state.next_state(action)
            print(cur_state)

    def print_policy(self, policy, values):
        print("Without passenger:")
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",pol[((i,j),self.state.passenger_loc,0)][0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(5):
            for j in range(5):
                print(values[((i,j),self.state.passenger_loc,0)], end = " ")
            print(" ")    
        print("With Passenger")  
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",pol[((i,j),(i,j),1)][0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(5):
            for j in range(5):
                print(values[((i,j),(i,j),1)], end = " ")
            print(" ")    
            


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
        self.q_table = {}
        for cur_state in self.state.possible_states:
            self.q_table[cur_state.val()] = np.zeros(6)
            
        for episode_num in range(NUM_EPISODES):
            print(episode_num)
            for state in self.state.possible_states:
                    
                random_val = random.uniform(0,1)
                action = ""
                action_num = 0
                if random_val < 0.1:
                    action_num = np.argmax(self.q_table[cur_state.val()])
                else:
                    action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
                
                action = self.index_to_action(action_num)
                # print(state.val(),action)
                reward = state.get_reward(action)
                next_state = state.next_state(action)
                self.q_table[state.val()][action_num]  =  self.q_table[state.val()][action_num] + ALPHA * (reward + GAMMA * np.max(self.q_table[next_state.val()]) - self.q_table[cur_state.val()][action_num])  
                # self.state = next_state

        return self.q_table
    



drop_loc = STATE_G                      
state = State((4,4), STATE_Y,False)
agent = Agent(state)
state.initialize()
# agent.learn_optimal_values()
# pol = agent.extract_policy(agent.new_values)
# agent.print_policy(pol, agent.new_values)
# agent.execute_policy(pol)

a,b = agent.policy_iteration()
agent.print_policy(a,b)