import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 
STATE_R = (0,0)
STATE_G = (0,4)
STATE_Y = (4,0)
STATE_B = (4,3)
GAMMA = float(sys.argv[1])     # discount factor
ALPHA = 0.25           # learning rate
EPISODE_SIZE = 500


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
        self.numbers = {}
    def val(self):
        return (self.taxi_loc,self.passenger_loc,int(self.passenger_sitting))

    def initialize(self) :
        grid_rows = 5
        grid_cols = 5

        if self.is_10_by_10:
            grid_rows = 10
            grid_cols = 10
        k = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                for a in range(grid_rows):
                    for b in range(grid_cols):
                        cur_state = State((i,j),(a,b),False,self.is_10_by_10)
                        self.possible_states.append(cur_state)
                        self.numbers[cur_state.val()] = k 
                        k = k + 1
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                cstate = State((i,j),(i,j),True,self.is_10_by_10)
                self.possible_states.append(cstate)       
                self.numbers[cstate.val()] = k 
                k = k + 1


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





# +-------------------------------------------------------------------------------------------------+

# Agent Class

class Agent:
    def __init__(self, state):
        self.state = state
        self.reward = 0

    def get_action(self):
        return np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.2,0.2,0.2,0.2,0.1,0.1])

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
        self.old_values = {}
        for state in self.state.possible_states:
            self.old_values[state.val()] = 0
        self.new_values = {}
        norm_vals = []
        while not converged:
            num_iter = num_iter + 1
            max_norm = -10000
            for state in self.state.possible_states:
                cur_state = State(state.taxi_loc, state.passenger_loc, state.passenger_sitting, self.state.is_10_by_10)
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
                            if pos_action == action:
                                value += 0.85*(reward_val + GAMMA * self.old_values[next_state.val()])
                            else:
                                value += 0.05*(reward_val + GAMMA * self.old_values[next_state.val()])
                                
                    if value > max_value:
                        max_value = value
    
                self.new_values[cur_state.val()] = max_value
                max_norm = max(max_norm, abs(self.old_values[cur_state.val()] - max_value))
            self.old_values = self.new_values.copy()
            norm_vals.append(max_norm)
            if max_norm < 1e-6:
                converged = True
            print(num_iter, max_norm)
            if num_iter > 5000 or converged:
                break
        return norm_vals


    def evaluate_from_values(self, policy, values):
        converged = False
        num_iter = 0
        old_values =  values
        new_values = {}
        
        while not converged:
            num_iter = num_iter + 1
            max_norm = -10000
            for state in self.state.possible_states:
                cur_state = State(state.taxi_loc, state.passenger_loc, state.passenger_sitting, self.state.is_10_by_10)
                max_value = -100000
                action = policy[cur_state.val()]
                value = 0
                if action == "pickup" :
                    reward_val = cur_state.get_reward(action)
                    next_state = cur_state.next_state(action)
                    value = reward_val + GAMMA * old_values[next_state.val()]
                elif action == "putdown":
                    reward_val = cur_state.get_reward(action)
                    next_state = cur_state.next_state(action)
                    if cur_state.taxi_loc == cur_state.drop_loc and cur_state.passenger_sitting:
                        value = reward_val
                    else:    
                        value = reward_val + GAMMA * old_values[next_state.val()] 
                else:
                    for pos_action in ["N","S","E","W"]:
                        reward_val = cur_state.get_reward(pos_action)
                        next_state = cur_state.next_state(pos_action)
                        if pos_action == action:
                            value += 0.85*(reward_val + GAMMA * old_values[next_state.val()])
                        else:
                            value += 0.05*(reward_val + GAMMA * old_values[next_state.val()])
                            
                if value > max_value:
                    max_value = value
                new_values[cur_state.val()] = max_value
                max_norm = max(max_norm, abs(old_values[cur_state.val()] - max_value))
            old_values = new_values.copy()

            if max_norm < 0.001:
                converged = True
            
            if num_iter > 500 or converged:
                break

        return new_values        



    def generate_random_policy(self):
        policy_result = {}
        for state in self.state.possible_states:
            # policy_result[state.val()] = np.random.choice(["N","S","E","W","pickup","putdown"],p=[0.2,0.2,0.2,0.2,0.1,0.1])
            policy_result[state.val()] = "N"
        return policy_result
    
    # A function to evaluate given policy by iterating through all possible states
    # Return: a dictionary of state_value    
    def evaluate_policy_iterative(self, policy, values, first):
        state_value = {}
        max_norm = -1
        for state in self.state.possible_states:
            cur_state = state
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
                    if pos_action == action:
                        value += 0.85*(reward_val + GAMMA * values[next_state.val()])
                    else:
                        value += 0.05*(reward_val + GAMMA * values[next_state.val()])
          
            state_value[cur_state.val()] = value
            if not first:
                max_norm = max(max_norm, abs(self.p_values[cur_state.val()] - max_value))
        return state_value, max_norm            
    
    def max_norm(self,values):
        max_norm = -1
        for state in self.state.possible_states:
            max_norm = max(max_norm, abs(self.p_values[state.val()] - values[state.val()]))
        return max_norm    
    
    def evaluate_policy_linear(self, policy):
        num_states = len(self.state.possible_states)
        A = np.zeros((num_states,num_states))
        C = np.zeros(num_states)
        X = np.zeros(num_states)
        for state in self.state.possible_states:
            cur_state = state
            max_value = -100000
            action = policy[cur_state.val()]
            value = 0
            if action == "pickup" :
                reward_val = cur_state.get_reward(action)
                i = self.state.numbers[cur_state.val()]
                next_state = cur_state.next_state(action)
                j = self.state.numbers[next_state.val()]
                A[i,j] = 1.00
                C[i] = reward_val 
                # value = reward_val + GAMMA * values[next_state.val()]
            elif action == "putdown":
                reward_val = cur_state.get_reward(action)
                i = self.state.numbers[cur_state.val()]
                next_state = cur_state.next_state(action)
                j = self.state.numbers[next_state.val()]
                A[i,j] = 1.00
                if cur_state.taxi_loc == cur_state.drop_loc and cur_state.passenger_sitting:
                    # value = reward_val
                    A[i,j] = 0.00
                C[i] = reward_val
            else:
                for pos_action in ["N","S","E","W"]:
                    reward_val = cur_state.get_reward(pos_action)
                    i = self.state.numbers[cur_state.val()]
                    next_state = cur_state.next_state(pos_action)
                    j = self.state.numbers[next_state.val()]
                   
                    if pos_action == action:
                        A[i,j] += 0.85
                        C[i] += 0.85*(reward_val)
                    else:
                        A[i,j] += 0.05
                        C[i] += 0.05*(reward_val)
        A = GAMMA * A
        I = np.identity(num_states)
        X = np.matmul(np.linalg.inv(I - A), C)
        state_values = {}
        for state in self.state.possible_states:
            state_values[state.val()] = X[self.state.numbers[state.val()]]
        return state_values

    def policy_iteration(self, first, linear): 
        policy = self.generate_random_policy()
        converged = False
        values = {}
        loss_vals = []
        for i in self.state.possible_states:
            values[i.val()] = 0  
        num_iter = 0
        prev_policy = policy.copy()
        while not converged:
            print(num_iter)
            prev_policy = policy.copy()
            if linear:
                values = self.evaluate_policy_linear(policy)
            else:    
                values = self.evaluate_from_values(policy, values)
            policy = self.extract_policy(values)
            if not first :
                loss_vals.append(self.max_norm(values))
            num_iter = num_iter + 1
            pchange = False
            for i in policy:
                if policy[i] != prev_policy[i]:
                    pchange = True
            if not pchange:
                converged = True
            if policy == prev_policy:
                converged = True
        return policy,values, loss_vals   

    def extract_policy(self, values):
        self.policy = {}
        for cur_state in self.state.possible_states:
            max_value = -100000
            next_state = None
            self.policy[cur_state.val()] = "U"
            for action in ["N","S","E","W","pickup","putdown"]:
                value = 0
                
                if action == "pickup" or action == "putdown":
                    reward_val = cur_state.get_reward(action)
                    next_state = cur_state.next_state(action)
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
                if value >= max_value:                 
                    max_value = value
                    self.policy[cur_state.val()] = action
            
        return self.policy

    def execute_policy(self, policy):
        cur_state = State(self.state.taxi_loc, self.state.passenger_loc, self.state.passenger_sitting, self.state.is_10_by_10)
        total_reward = 0
        # print(cur_state)
        print(cur_state.val())
        num_iter = 0
        while not cur_state.finished or num_iter > 100:
            num_iter = num_iter + 1
            action = policy[cur_state.val()]
            print("Action Intended:",action)
            action = self.state.choose_action(action)
            print("Action Succeeded:", action)
            cur_reward = cur_state.get_reward(action)
            print("Reward:", cur_reward)
            total_reward += cur_reward
            print("Total Reward:", total_reward)
            print("+--------------------------+")
            cur_state = cur_state.next_state(action)
            # print(cur_state)
            print(cur_state.val())

    def print_policy(self, policy, values):
        print("Without passenger:")
        print(" ","Policy")     
        grid_rows = 5
        grid_cols = 5
        if self.state.is_10_by_10:
            grid_rows = 10
            grid_cols = 10   
        for i in range(grid_rows):
            for j in range(grid_cols):
                print("     ",policy[((i,j),self.state.passenger_loc,0)][0], end = " ")
            print(" ")    
        # print(" ", "Values")
        # for i in range(grid_rows):
        #     for j in range(grid_cols):
        #         print(values[((i,j),self.state.passenger_loc,0)], end = " ")
        #     print(" ")    
        print("With Passenger")  
        print(" ","Policy")        
        for i in range(grid_rows):
            for j in range(grid_cols):
                print("     ",policy[((i,j),(i,j),1)][0], end = " ")
            print(" ")    
        # print(" ", "Values")
        # for i in range(grid_rows):
        #     for j in range(grid_cols):
        #         print(values[((i,j),(i,j),1)], end = " ")
        #     print(" ")    
            
    def print_policy_q(self, qvalues):
        print("Without passenger:")
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",self.index_to_action(np.argmax(qvalues[((i,j),self.state.passenger_loc,0)]))[0], end = " ")
            print(" ")    
        # print(" ", "Values")
        # for i in range(5):
        #     for j in range(5):
        #         print(np.max(qvalues[((i,j),self.state.passenger_loc,0)]), end = " ")
        #     print(" ")    
        print("With Passenger")  
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",self.index_to_action(np.argmax(qvalues[((i,j),(i,j),1)]))[0], end = " ")
            print(" ")    
        # print(" ", "Values")
        # for i in range(5):
        #     for j in range(5):
        #         print(np.max(qvalues[((i,j),(i,j),1)]), end = " ")
        #     print(" ")    
            


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
            return "U"   

    def q_learning(self, decay, num_episodes):
        """N, S, E, W, pickup, Drop"""
        self.q_table = {}
        for cur_state in self.state.possible_states:
            self.q_table[cur_state.val()] = np.zeros(6)
        epsilon = 0.1    
        num_iter = 0
        discounted_rewards = []
        for episode_num in tqdm(range(num_episodes)):
            step = 0
            # print(episode_num)
            state = np.random.choice(self.state.possible_states)
            # print(state.val())
            total_reward = 0
            while True:
                random_val = np.random.uniform(0,1)
                action = ""
                action_num = 0
                if random_val < epsilon:
                    action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
                else:
                    action_num = np.argmax(self.q_table[state.val()])
                
                action = self.index_to_action(action_num)
                action = state.choose_action(action)
                reward = state.get_reward(action)
                next_state = state.next_state(action)
                self.q_table[state.val()][action_num]  =  (1 - ALPHA) * self.q_table[state.val()][action_num] + ALPHA * (reward + GAMMA * np.max(self.q_table[next_state.val()]) )  
                state = State(next_state.taxi_loc, next_state.passenger_loc, next_state.passenger_sitting, next_state.is_10_by_10)
                step = step  + 1
                num_iter = num_iter + 1
                total_reward = GAMMA * total_reward + reward
                if step > EPISODE_SIZE or next_state.finished:
                    break
            if decay:
                epsilon = epsilon / num_iter
            # discounted_rewards.append(total_reward)
        return self.q_table
    def select_action(self, cur_state,epsilon):
        random_val = np.random.uniform(0,1)
        action_num = 0
        if random_val < epsilon:
            action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
        else:
            action_num = np.argmax(self.sarsa_table[cur_state.val()])
        
        action = self.index_to_action(action_num)
        action = cur_state.choose_action(action)
        return action, action_num

    def sarsa_learning(self, decay, num_episodes):
        """N, S, E, W, pickup, Drop"""
        self.sarsa_table = {}
        for cur_state in self.state.possible_states:
            self.sarsa_table[cur_state.val()] = np.zeros(6)
        epsilon = 0.1    
        num_iter = 0
        discounted_rewards = []
        for episode_num in tqdm(range(num_episodes)):
            step = 0
            # print(episode_num)
            state_f = np.random.choice(self.state.possible_states)
            action_f, action_num_f = self.select_action(state_f, epsilon)
            total_reward = 0
            while True:
                reward = state_f.get_reward(action_f)
                state_s = state_f.next_state(action_f)
                action_s, action_num_s = self.select_action(state_s, epsilon)
                self.sarsa_table[state_f.val()][action_num_f]  =  (1 - ALPHA) * self.sarsa_table[state_f.val()][action_num_f] + ALPHA * (reward + GAMMA * self.sarsa_table[state_s.val()][action_num_s])  
                state_f = State(state_s.taxi_loc, state_s.passenger_loc, state_s.passenger_sitting, state_s.is_10_by_10)
                action_f = action_s
                action_num_f = action_num_s
                step = step  + 1
                num_iter = num_iter  + 1
                total_reward = GAMMA * total_reward + reward
                if step > EPISODE_SIZE or state_s.finished:
                    break
            if decay:
                epsilon = epsilon / num_iter
            discounted_rewards.append(total_reward)
        return self.sarsa_table
    def get_d_rewards(self, method, num_episodes, q_vals = None):
        if method == "q":
            q_vals = self.q_learning(False, num_episodes)
        elif method == "sarsa":
            q_vals = self.sarsa_learning(False, num_episodes)
        elif method == "q_delay":
            q_vals = self.q_learning(True, num_episodes)
        elif method == "sarsa_delay":
            q_vals = self.sarsa_learning(True, num_episodes)
        d_rewards = 0
        num_instances = 100
        for i in range(num_instances):
            # print(i)
            d_factor = GAMMA
            total_reward = 0
            state = self.state
            for j in range(EPISODE_SIZE):
                action_num = np.argmax(q_vals[state.val()])
                action = self.index_to_action(action_num)
                action = state.choose_action(action)
                reward = state.get_reward(action)
                next_state = state.next_state(action)
                state = State(next_state.taxi_loc, next_state.passenger_loc, next_state.passenger_sitting, next_state.is_10_by_10)
                total_reward = total_reward + d_factor * reward
                # print(reward,total_reward)
                d_factor = d_factor * GAMMA
                if next_state.finished:
                    break
            d_rewards += total_reward
        return d_rewards / num_instances


def plot(x_vals, y_vals,title,x_label,y_label,opt):
    plt.title(title, c = 'b')
    plt.xlabel(x_label, c = 'purple')
    plt.ylabel(y_label, c = 'purple')
    plt.plot(x_vals, y_vals, c = 'r')
    plt.savefig("./../plots/" + opt + '.png')
    print("Plot saved to "+"./../plots/" + opt + '.png')
    plt.close()



drop_loc = STATE_G
passenger_loc = STATE_Y                      
taxi_loc = STATE_R
state = State(taxi_loc, passenger_loc, False, False)
agent = Agent(state)
state.initialize()

# norm_vals = agent.learn_optimal_values()
# policy = agent.extract_policy(agent.new_values)
# agent.print_policy(policy, agent.new_values)
# agent.execute_policy(policy)

# plot(1 + np.array(range(len(norm_vals))), norm_vals, "Plot of Iteration number vs Max-Norm Values for gamma = "+ str(GAMMA), "Iteration Number", "Max-Norm Values", "optimal_values("+ str(GAMMA) +")")

# policy_p_iter,values_p_iter, _ = agent.policy_iteration(True, False)
# agent.p_values = values_p_iter
# agent.print_policy(policy_p_iter, values_p_iter)
# # agent.execute_policy(policy_p_iter)

# x, y, ploss = agent.policy_iteration(False, False)
# plot(1 + np.array(range(len(ploss))), ploss, "Plot of Iteration number vs Policy Loss Values for gamma = "+ str(GAMMA), "Iteration Number", "Policy Loss", "Policy Loss("+ str(GAMMA) +")")




# policy_p_iter,values_p_iter, _ = agent.policy_iteration(True, True)
# agent.p_values = values_p_iter
# agent.print_policy(policy_p_iter, values_p_iter)
# # agent.execute_policy(policy_p_iter)

# x, y, ploss = agent.policy_iteration(False, True)
# plot(1 + np.array(range(len(ploss))), ploss, "Plot of Iteration num. vs Policy Loss(Linear method) for gamma = "+ str(GAMMA), "Iteration Number", "Policy Loss", "Policy Loss Linear("+ str(GAMMA) +")")






# q_vals, d_rewards = agent.q_learning(False)
# agent.print_policy_q(q_vals)    
# plot(1 + np.array(range(len(d_rewards))), d_rewards, "Plot of Iteration num. vs Discounted rewards for gamma = "+ str(GAMMA), "Iteration Number", "Discounted Rewards", "Discouted Rewards("+ str(GAMMA) +")")



# q_vals, d_rewards = agent.sarsa_learning(False)
# agent.print_policy_q(q_vals)    
# plot(1 + np.array(range(len(d_rewards))), d_rewards, "Plot of num_iter. vs Discounted rewards (SARSA", "Iteration Number", "Discounted Rewards", "Sarsa("+ str(GAMMA) +")")



# q_vals = agent.q_learning(True)
# agent.print_policy_q(q_vals)    


for method in ["q", "q_delay", "sarsa", "sarsa_delay"]:
    vals = []
    for i in range(1,10):
        vals.append(agent.get_d_rewards(method, i*1000))
        print("Reward = ", vals[-1])
        plot(1000 + 1000* np.array(range(len(vals))), vals, "Plot of num_episodes vs Discounted rewards (" + method + ")", "Number of episodes", "Sum of Discounted Rewards", "Plot + "+ method)

# sarsa_vals = agent.sarsa_learning(False)
# agent.print_policy_q(sarsa_vals)



