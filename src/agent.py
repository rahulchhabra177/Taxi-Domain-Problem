from state import *
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
            print(num_iter,",", max_norm)

            if max_norm < 0.001:
                converged = True
            
            if num_iter > 500 or converged:
                break



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
                        # if next_state == cur_state:
                        #     continue
                        if pos_action == action:
                            value += 0.85*(reward_val + GAMMA * old_values[next_state.val()])
                        else:
                            value += 0.05*(reward_val + GAMMA * old_values[next_state.val()])
                            
                if value > max_value:
                    max_value = value
            # print(max_value)   
            # if cur_state.taxi_loc == (0,4):
            #     print(cur_state.val())
            #     print(max_value)
            #     print(self.old_values[cur_state.val()])     
                new_values[cur_state.val()] = max_value
            # print(cur_state.val())
                max_norm = max(max_norm, abs(old_values[cur_state.val()] - max_value))
        
            # print(abs(self.old_values[cur_state.val()] - max_value),self.old_values[cur_state.val()], max_value)
            # print(cur_state.val(), max_value)               
            old_values = new_values.copy()
            # print(num_iter,",", max_norm)

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
            # print(action)
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
          
            state_value[cur_state.val()] = value
            if not first:
                max_norm = max(max_norm, abs(self.p_values[cur_state.val()] - max_value))
        return state_value, max_norm            
    # a function to evaluate a given policy by using numpy linear algebra to solve the linear equation
    
    def max_norm(self,values):
        max_norm = -1
        for state in self.state.possible_states:
            max_norm = max(max_norm, abs(self.p_values[state.val()] - values[state.val()]))
        return max_norm    
    
    def evaluate_policy_linear_equation(self, policy):
        A = np.zeros((len(self.state.possible_states),len(self.state.possible_states)))
        b = np.zeros(len(self.state.possible_states))
    
    def policy_iteration(self, first): 
        policy = self.generate_random_policy()
        converged = False
        values = {}
        for i in self.state.possible_states:
            values[i.val()] = 0  
        num_iter = 0
        prev_policy = policy.copy()
        while not converged:
            prev_policy = policy.copy()
            values = self.evaluate_from_values(policy, values)
            policy = self.extract_policy(values)
            # print(num_iter)
            if not first :
                print(num_iter, ",",self.max_norm(values))
            num_iter = num_iter + 1
            pchange = False
            # self.print_policy(policy, values)
            for i in policy:
                if policy[i] != prev_policy[i]:
                    pchange = True
            if not pchange:
                # print("N:",num_iter)
                converged = True
            if policy == prev_policy:
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
                # print(cur_state.val(), action, value)
                if value >= max_value:                 
                    max_value = value
                    self.policy[cur_state.val()] = action
            
        return self.policy

    def execute_policy(self, policy):
        cur_state = State(self.state.taxi_loc, self.state.passenger_loc, self.state.passenger_sitting, self.state.is_10_by_10)
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
        grid_rows = 5
        grid_cols = 5
        if self.state.is_10_by_10:
            grid_rows = 10
            grid_cols = 10   
        for i in range(grid_rows):
            for j in range(grid_cols):
                print("     ",policy[((i,j),self.state.passenger_loc,0)][0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(grid_rows):
            for j in range(grid_cols):
                print(values[((i,j),self.state.passenger_loc,0)], end = " ")
            print(" ")    
        print("With Passenger")  
        print(" ","Policy")        
        for i in range(grid_rows):
            for j in range(grid_cols):
                print("     ",policy[((i,j),(i,j),1)][0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(grid_rows):
            for j in range(grid_cols):
                print(values[((i,j),(i,j),1)], end = " ")
            print(" ")    
            
    def print_policy_q(self, qvalues):
        print("Without passenger:")
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",self.index_to_action(np.argmax(qvalues[((i,j),self.state.passenger_loc,0)]))[0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(5):
            for j in range(5):
                print(np.max(qvalues[((i,j),self.state.passenger_loc,0)]), end = " ")
            print(" ")    
        print("With Passenger")  
        print(" ","Policy")        
        for i in range(5):
            for j in range(5):
                print("     ",self.index_to_action(np.argmax(qvalues[((i,j),(i,j),1)]))[0], end = " ")
            print(" ")    
        print(" ", "Values")
        for i in range(5):
            for j in range(5):
                print(np.max(qvalues[((i,j),(i,j),1)]), end = " ")
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
            return "U"   

    def q_learning(self, decay):
        """N, S, E, W, pickup, Drop"""
        self.q_table = {}
        for cur_state in self.state.possible_states:
            self.q_table[cur_state.val()] = np.zeros(6)
        epsilon = 0.1    
        num_iter = 0
        for episode_num in range(NUM_EPISODES):
            print(episode_num)
            step = 0
            state = np.random.choice(self.state.possible_states)
            while True:
                random_val = random.uniform(0,1)
                action = ""
                action_num = 0
                if random_val < 5*epsilon:
                    action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
                else:
                    action_num = np.argmax(self.q_table[cur_state.val()])
                
                action = self.index_to_action(action_num)
                action = state.choose_action(action)
                reward = state.get_reward(action)
                next_state = state.next_state(action)
                self.q_table[state.val()][action_num]  =  (1 - ALPHA) * self.q_table[state.val()][action_num] + ALPHA * (reward + GAMMA * np.max(self.q_table[next_state.val()]) )  
                state = next_state
                step = step  + 1
                num_iter = num_iter + 1
                if step > EPISODE_SIZE or state.finished:
                    break
            if decay:
                epsilon = epsilon / num_iter

        return self.q_table
    def select_action(self, cur_state,epsilon):
        random_val = random.uniform(0,1)
        action_num = 0
        if random_val < epsilon:
            action_num = np.random.choice([0,1,2,3,4,5],p=[1/6,1/6,1/6,1/6,1/6,1/6])
        else:
            action_num = np.argmax(self.sarsa_table[cur_state.val()])
        
        action = self.index_to_action(action_num)
        action = cur_state.choose_action(action)
        return action, action_num

    def sarsa_learning(self, decay):
        """N, S, E, W, pickup, Drop"""
        self.sarsa_table = {}
        for cur_state in self.state.possible_states:
            self.sarsa_table[cur_state.val()] = np.zeros(6)
        epsilon = 0.1    
        num_iter = 0
        for episode_num in range(NUM_EPISODES):
            step = 0
            state_f = np.random.choice(self.state.possible_states)
            action_f, action_num_f = self.select_action(state_f, epsilon)
            while True:
                reward = state_f.get_reward(action_f)
                state_s = state_f.next_state(action_f)
                action_s, action_num_s = self.select_action(state_s, epsilon)
                self.sarsa_table[state_f.val()][action_num_f]  =  (1 - ALPHA) * self.sarsa_table[state_f.val()][action_num_f] + ALPHA * (reward + GAMMA * self.sarsa_table[state_s.val()][action_num_s])  
                state_f = State(state_s.taxi_loc, state_s.passenger_loc, state_s.passenger_sitting)
                action_f = action_s
                action_num_f = action_num_s
                step = step  + 1
                num_iter = num_iter  + 1
                if step > EPISODE_SIZE or state_f.finished:
                    break
            if decay:
                epsilon = epsilon / step

        return self.sarsa_table


