import gymnasium as gym
from gymnasium import spaces
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy, NMF
from surprise.model_selection import train_test_split
import torch
import pandas as pd
from sklearn.utils import shuffle
#from ray.rllib.env.multi_agent_env import MultiAgentEnv

#np.random.seed(0)

def to_data(atensor):
    sparse_state = atensor.to_sparse()
    df = pd.DataFrame({'user':sparse_state.indices()[0].tolist(),'item':sparse_state.indices()[1].tolist(),'rating':sparse_state.values().tolist()})
    #print(df)
    #df = shuffle(df)
    #print(df)
    #return
    reader = Reader(rating_scale=(1, 3))  # Assuming the rating scale is 1-5
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    #trainset,testset = train_test_split(data,test_size=0.4)
    #return trainset,testset
    return data.build_full_trainset()

def to_test(atensor,u):
    sparse_state = atensor.to_sparse()
    df = pd.DataFrame({'user':[u]*len(sparse_state.values()),'item':sparse_state.indices()[0].tolist(),'rating':sparse_state.values().tolist()})
    reader = Reader(rating_scale=(1, 3))  # Assuming the rating scale is 1-5
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    data = data.build_full_trainset()
    return data.build_testset()

#class MultiAgentStationaryImplicit(MultiAgentEnv):
class MultiAgentStationaryImplicit(gym.Env):
    def __init__(self, cat_size=3, num_items=10, num_users=9, num_players=None, player_indices=None, memory_max=3, memory_factor = 30, q=0.3, privacy_constant=1.2,utility_constant=1.08,weight=0.5,single_done=False):
        super(MultiAgentStationaryImplicit, self).__init__()
        self.cat_size = cat_size
        self.num_items = num_items
        self.num_users = num_users
        self.memory_max = memory_max
        self.memory_factor = memory_factor
        self.q = q
        self.privacy_constant = privacy_constant
        self.utility_constant = utility_constant
        self.weight = weight
        self.single_done = single_done

        self.num_players = num_players if num_players else num_users

        self.player_indices = player_indices if player_indices else list(range(self.num_players))

        self.agents = [str(i) for i in self.player_indices]

        self.observation_space = spaces.Dict({
            'underlying':spaces.MultiDiscrete(np.array([3]*self.num_items)),
            'revealing':spaces.MultiBinary(self.num_items),
            "categories":spaces.MultiBinary(self.cat_size),
            "N":spaces.Discrete(self.memory_max)
        })
        self.observation_spaces = dict(zip(self.agents,[self.observation_space]*len(self.agents)))
        # self.action_space = spaces.Dict({
        #       "categories":spaces.MultiBinary(self.cat_size),
        #       "N":spaces.Discrete(self.memory_max,start=1)}
        #     )
        #self.action_space = spaces.MultiDiscrete([2] * self.cat_size + [self.memory_max])
        # Original MultiDiscrete action space
        self.multi_discrete_space = spaces.MultiDiscrete([2] * self.cat_size + [self.memory_max])

        # Total number of actions for Discrete space
        self.n_actions = np.prod([2] * self.cat_size + [self.memory_max])

        # Flattened action space as Discrete for DQN
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_spaces = dict(zip(self.agents,[self.action_space]*len(self.agents)))

        self.current_step=0
        self.max_steps = 50
        self.category2item = np.array([[1,0,0],
                                       [1,0,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,0,1],
                                       [0,0,1],
                                       [0,0,1],
                                       [0,0,1]])
        # self.underlying = np.array([    [1, 2, 1, 1, 1, 1, 2, 2, 2, 2], # last 4 items for testing
        #                                 [3, 2, 3, 3, 3, 3, 2, 2, 2, 2],
        #                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                                 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #                                 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #                                 [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
        #                                 [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
        #                                 [2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
        #                                 [2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
        #                                 [3, 1, 3, 3, 3, 3, 1, 1, 1, 1]])
                                        #[3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
                                        #[3, 1, 3, 3, 3, 3, 1, 1, 1, 1],   # last 3 user constantly sharing
                                        #[3, 2, 3, 3, 3, 3, 2, 2, 2, 2],
                                        #[1, 2, 1, 1, 1, 1, 2, 2, 2, 2],]) 
        self.underlying = np.array([    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 2, 1, 1, 1, 1, 2, 2, 2, 2],
                                        [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
                                        [2, 1, 2, 2, 2, 2, 1, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
                                        [3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
                                        [3, 2, 3, 3, 3, 3, 2, 2, 2, 2],
                                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
        # self.categories = np.array([[1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,0],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,0,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1]],dtype=np.int8)
        self.categories = np.array([[1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1],
                                    [1,1,1]],dtype=np.int8)
        self.N = np.random.randint(0,self.memory_max,(self.num_users,),dtype=np.int32)*self.memory_factor+1
        #self.N[-1] = (self.memory_max-1)*self.memory_factor+1
        #self.N = np.ones((self.num_users,),dtype=np.int32)*10
        self.revealing2 = np.random.choice([0, 1], size=(self.num_users, self.num_items), p=[0.7, 0.3]) #to store the self.revealing after applying category filter
        self.revealing = self.revealing2.copy()  #to keep track of revealing evolution with time
        # self.revealing[-1][-1] = 1  # if we want to keep one player sharing everything
        # self.revealing2[-1][-1] = 1
        # self.revealing[-2] = np.array([1]*self.num_items,dtype=np.int8)  # if we want to keep one player sharing everything
        # self.revealing2[-2] = np.array([1]*self.num_items,dtype=np.int8)
        self.reward = dict(zip(self.agents,[0.0]*self.num_players))
        self.utility = dict(zip(self.agents,[0.0]*self.num_players))
        self.privacy = dict(zip(self.agents,[0.0]*self.num_players))
        #self.C_u = np.zeros(self.cat_size,dtype=np.int8)

    def decode_action(self, action):
        """
        Decode a Discrete action back into the original MultiDiscrete components.
        """
        # Convert the single discrete action into a multi-dimensional action
        multi_discrete_action = np.unravel_index(action, self.multi_discrete_space.nvec)
        return multi_discrete_action

    def get_observation_space(self,agent_id=None):
        return self.observation_space

    def get_action_space(self,agent_id=None):
        return self.action_space

    def _get_obs(self,i=-1):
        return {'underlying':self.underlying[i]-1,'revealing': self.revealing[i],'categories':self.categories[i],'N':(self.N[i]-1)//self.memory_factor}

    def step(self, action_dict):
        #p = self.player_indices
        delta_I_r = []

        #print('Population move')
        # Population movement
        for i in range(self.num_users):
          if i in self.player_indices:  # Player movement
            delta_I_u_r= self.player_move(i,action_dict[str(i)])
          # elif i==self.num_users-1:
          #   delta_I_u_r = np.array([0]*self.num_items)
          else:
            delta_I_u_r = self.auto_move(i)
          #print(delta_I_u_r)
          delta_I_r.append(delta_I_u_r)

        #print('Check full')
        for i in self.player_indices:
          delta_I_u_r = delta_I_r[i]
          if sum(delta_I_u_r) == 0:  # no generated data, only happens if revealing vector is already full
              print('Empty')
              print('Full at step %d',self.current_step)
              observations = {}
              for j in range(len(self.player_indices)):
                observations[self.agents[j]] = self._get_obs(self.player_indices[j])
              if self.single_done:
                terminated = True
                truncated = False
              else:
                terminated = dict(zip(self.agents,[True]*self.num_players))
                truncated = dict(zip(self.agents,[False]*self.num_players))
                terminated['__all__'] = True
                truncated['__all__'] = False
              return observations, self.reward, terminated, truncated, {}

        # System train
        self.system_train()

        # Assign a reward
        for p in self.player_indices:
          delta_I_u = self.underlying[p]*delta_I_r[p]
          reward = self.reward_function(delta_I_u,p)
          self.reward[str(p)]=reward
        #reward = 0.1
        self.current_step += 1

        for i in range(1,self.num_users):
          delta_i_u_r = delta_I_r[i]
          self.transition2(i,delta_i_u_r)

        observations = {}
        for i in self.player_indices:
          observations[str(i)] = self._get_obs(i)
        # Check if the game is done
        done = self.current_step >= self.max_steps
        if self.single_done:
                terminated = done
                truncated = False
        else:
                terminated = dict(zip(self.agents,[done]*self.num_players))
                truncated = dict(zip(self.agents,[False]*self.num_players))
                terminated['__all__'] = done
                truncated['__all__'] = False
        return observations, self.reward, terminated, truncated, {}

    def category_filter(self,A,C):
        mask = (np.matmul(self.category2item,C) == 0)
        A[mask] = 0
        return A

    def auto_move(self,i):
        A = self.revealing[i]
        C = self.categories[i]
        N1 = N2 = self.N[i]
        A = self.transition(A,C,N1,N2)
        delta_I_u_r = self.generate_new_revealing(A)
        self.revealing[i] = A
        self.revealing2[i] = self.category_filter(A,C)
        return delta_I_u_r

    def player_move(self,i,action):
        #print(action)
        action = self.decode_action(action)
        # Player movement
        A = self.revealing[i]
        N1 = self.N[i]

        C = action[:self.cat_size]
        N2 = action[self.cat_size]*self.memory_factor+1
        A, N2 = self.transition(A,C,N1,N2)
        delta_I_u_r = self.generate_new_revealing(A,player=True)
        self.categories[i] = C
        self.N[i]=N2
        self.revealing[i] = A
        self.revealing2[i] = self.category_filter(A,C)
        return delta_I_u_r

    def generate_new_revealing(self,A,player=True): # A the revealing vector of a player
        delta_I_u_r = np.zeros_like(A)
        mask = (A == 0)
        delta_I_u_r[mask] = np.random.choice([0, 1], size=mask.sum(), p=[1 - self.q, self.q])
        if player and sum(delta_I_u_r)==0 and sum(A)!=len(A) and self.q != 0:
          delta_I_u_r[np.random.choice(np.where(mask)[0])] = 1
        return delta_I_u_r

    def transition(self,A, C, N1, N2):
        # Ensure N2 does not exceed N1 + 1
        N2 = min(N2, N1 + 1)
        #print(N1,N2)
        # Calculate the probability to keep 1
        prob = (1 - (1 - self.q)**(N2-1)) / (1 - (1 - self.q)**(N1))
        #print(prob)
        # mask = (np.matmul(self.category2item,C) == 0)
        # A[mask] = 0
        A0 = A.copy()
        mask2 = (A == 1)
        A[mask2] = np.random.choice([1, 0], size=mask2.sum(), p=[prob, 1-prob])
        if (A==A0).all() and sum(A) == len(A) and prob<1:
          A[np.random.choice(np.where(mask2)[0])] = 0
        return A, N2

    def transition2(self,u,delta_I_u_r):
        self.revealing[u] += delta_I_u_r
        self.revealing2[u] = self.category_filter(self.revealing[u],self.categories[u])

    def reward_function(self, delta_I,p):
        #print(self.revealing[-1])
        #print(all_data[-1])
        predictions = self.system_pred(delta_I,p)

        utility = self.utility_function(predictions)
        self.utility[str(p)] = utility
        #utility = 0.2
        privacy = self.privacy_function(self.revealing2[-1])
        self.privacy[str(p)] = privacy
        #privacy = 0.2
        reward = utility*self.weight+privacy*(1-self.weight)  # Sum of all elements in the state
        return reward

    def system_train(self):
        all_data = self.underlying*self.revealing2
        train_set = to_data(torch.from_numpy(all_data))
        # Initialize and train the FunkSVD model
        # self.algo = NMF(n_factors=2,reg_pu=0, reg_qi=0,lr_bu=0.005, lr_bi=0.005,n_epochs=1000)
        self.algo = NMF(n_factors=2,n_epochs=2000)
        self.algo.fit(train_set)

    def system_pred(self,delta_I,p):
        delta_I = torch.from_numpy(delta_I)
        test_set = to_test(delta_I,p)
        # Evaluate on test set
        predictions_u = self.algo.test(test_set)
        #predictions_u = 0
        return predictions_u

    def utility_function(self,prediction):
        return self.utility_constant-accuracy.rmse(prediction, verbose=False)

    def privacy_function(self,revealing_vector):
        return self.privacy_constant*(1-np.mean(revealing_vector))

    def render(self, mode='human', message=True):
        # print(f"Step: {self.current_step}, State: {self.revealing[-1]*self.underlying[-1]},{self.N[-1]}")
        if message:
            print(f"Step: {self.current_step}, State: {self._get_obs()}, Reward: {self.utility}, {self.privacy}")
        return self.reward, self.utility, self.privacy

    def reset(self,seed=None,options=None):
        # self.categories = np.array([[1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1],
        #                             [1,1,1]],dtype=np.int8)
        #self.categories = np.random.randint(0,1,(self.num_users,3),dtype=np.int8)
        self.categories = np.random.choice([0, 1], size=(self.num_users, 3), p=[0.2, 0.8]).astype(np.int8)
        # self.N = np.random.randint(2,self.memory_max+1,(self.num_users,),dtype=np.int32)*self.memory_factor
        # self.N[-1] = (self.memory_max-1)*self.memory_factor+1
        #self.N = np.random.randint(0,self.memory_max,(self.num_users,),dtype=np.int32)*self.memory_factor+5
        self.N = np.random.choice(list(range(self.memory_max)), size=(self.num_users, ), p=[0.0, 0.1, 0.9]).astype(np.int32)*self.memory_factor+1
        # self.revealing2 = np.zeros((self.num_users,self.num_items),dtype=np.int8)
        # self.revealing = np.zeros((self.num_users,self.num_items),dtype=np.int8)
        self.revealing2 = np.random.choice([0, 1], size=(self.num_users, self.num_items), p=[0.7, 0.3]) #to store the self.revealing after applying category filter
        self.revealing = self.revealing2.copy()  #to keep track of revealing evolution with time
        # self.revealing[-1][-1] = 1  # if we want to keep one player sharing everything
        # self.revealing2[-1][-1] = 1
        # self.revealing[-1] = np.array([1]*self.num_items,dtype=np.int8)
        # self.revealing2[-1] = np.array([1]*self.num_items,dtype=np.int8)
        #observation = {'underlying':self.underlying[-1]-1,'revealing': self.revealing[-1],'categories':self.categories[-1],'N':self.N[-1]-1}
        observations = {}
        for i in range(self.num_players):
          observations[self.agents[i]] = self._get_obs(self.player_indices[i])
        # observation = self._get_obs()

        self.current_step = 0
        return observations,{}
    

class MultiAgentGroup(MultiAgentStationaryImplicit):
    def __init__(self, cat_size=3, num_items=10, num_groups=2, num_players=None,player_indices=None, agent_per_group=1, memory_max=3, memory_factor=30, q=0.3, privacy_constant=1.2, utility_constant=1.08, weight=0.6, single_done=False):
        self.cat_size = cat_size
        self.num_items = num_items
        self.num_groups = num_groups
        self.agent_per_group = agent_per_group
        self.num_users = num_groups*agent_per_group
        self.memory_max = memory_max
        self.memory_factor = memory_factor
        self.q = q
        self.privacy_constant = privacy_constant
        self.utility_constant = utility_constant
        self.weight = weight
        self.single_done = single_done
        self.current_step=0
        self.max_steps = 50

        # Multi Agent Setup
        self.num_players = num_players if num_players else self.num_users
        self.player_indices = player_indices if player_indices else list(range(self.num_players))
        self.agents = [str(i) for i in self.player_indices]

        # Observation Space
        self.observation_space = spaces.Dict({
            'underlying':spaces.MultiDiscrete(np.array([3]*self.num_items)),
            'revealing':spaces.MultiBinary(self.num_items),
            "categories":spaces.MultiBinary(self.cat_size),
            "N":spaces.Discrete(self.memory_max)
        })
        self.observation_spaces = dict(zip(self.agents,[self.observation_space]*len(self.agents)))

        # Action Space
        # self.action_space = spaces.MultiDiscrete([2] * self.cat_size + [self.memory_max])
        self.multi_discrete_space = spaces.MultiDiscrete([2] * self.cat_size + [self.memory_max])
        self.n_actions = np.prod([2] * self.cat_size + [self.memory_max])
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_spaces = dict(zip(self.agents,[self.action_space]*len(self.agents)))

        # Category to item mapping
        self.category2item = np.array([[1,0,0],
                                       [1,0,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,1,0],
                                       [0,0,1],
                                       [0,0,1],
                                       [0,0,1],
                                       [0,0,1]])
        
        # Initialization
        # self.underlying_candidates = np.array([   
        #                                 [3, 2, 3, 3, 3, 3, 2, 2, 2, 2],
        #                                 [2, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        #                                 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #                                 [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
        #                                 [2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
        #                                 [3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
        #                                 [1, 2, 1, 1, 1, 1, 2, 2, 2, 2],]) 
        self.underlying_candidates = np.array([ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 2, 1, 1, 1, 1, 2, 2, 2, 2],
                                        [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
                                        [2, 1, 2, 2, 2, 2, 1, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
                                        [3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
                                        [3, 2, 3, 3, 3, 3, 2, 2, 2, 2],
                                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
        # self.underlying_candidates= np.array([  
        #                                 [3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
        #                                 [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
        #                                 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #                                 ])
        # self.underlying_candidates= np.array([  [1, 2, 1, 1, 1, 1, 2, 2, 2, 2],
        #                                         [1, 3, 1, 1, 1, 1, 3, 3, 3, 3],
        #                                         [3, 1, 3, 3, 3, 3, 1, 1, 1, 1],
        #                                         [3, 2, 3, 3, 3, 3, 2, 2, 2, 2]])
        self.underlying = np.vstack([[self.underlying_candidates[i]]*self.agent_per_group for i in range(self.num_groups)])
        self.categories = np.random.randint(0,1,(self.num_users,3),dtype=np.int8)
        self.N = np.random.randint(0,self.memory_max,(self.num_users,),dtype=np.int32)*self.memory_factor+1
        self.revealing2 = np.random.choice([0, 1], size=(self.num_users, self.num_items), p=[0.7, 0.3]) #to store the self.revealing after applying category filter
        self.revealing = self.revealing2.copy()  #to keep track of revealing evolution with time
        self.reward = dict(zip(self.agents,[0.0]*self.num_players))
        self.utility = dict(zip(self.agents,[0.0]*self.num_players))
        self.privacy = dict(zip(self.agents,[0.0]*self.num_players))

class MultiAgentGroupStageThree(MultiAgentGroup):
    def __init__(self, robot=None,cat_size=3, num_players=None, player_indices=None, num_items=10, num_groups=2, agent_per_group=1, memory_max=3, memory_factor=30, q=0.3, privacy_constant=1.2, utility_constant=1.08, weight=0.6, single_done=False):
        super().__init__(cat_size=cat_size, num_items=num_items,num_players=num_players, player_indices=player_indices,num_groups=num_groups, agent_per_group=agent_per_group, memory_max=memory_max, memory_factor=memory_factor, q=q, privacy_constant=privacy_constant, utility_constant=utility_constant, weight=weight, single_done=single_done)
        self.robot = robot

    def auto_move(self,i):
        obs = self._get_obs(i)
        action,_ = self.robot.predict(obs,deterministic=True)
        return self.player_move(i,action)
        # A = self.revealing[i]
        # C = self.categories[i]
        # N1 = N2 = self.N[i]
        # A = self.transition(A,C,N1,N2)
        # delta_I_u_r = self.generate_new_revealing(A)
        # self.revealing[i] = A
        # self.revealing2[i] = self.category_filter(A,C)
        # return delta_I_u_r
