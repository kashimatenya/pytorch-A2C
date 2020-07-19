from datetime import datetime
import os
import shutil 
from statistics import median, mean

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import A2CAgentCreater, SeparateA2CModel, SharedA2CModel, SequentialMemories

from cartpole import CartPoleV0, ValueNet, PolicyNet, DualNet

from multiple import MultipleEnvironments, MultipleSequentialMemories

from log import DisplayLog, FileLog, Logs, aggregate_stats


#settings
max_epochs = 100
training_per_epoch = 100
training_environments_num = 8

test_interval = 2
episodes_to_test = 100

model_type = "shared"

use_gpu = False

#size parameter for memory
max_sequencies_in_training_memory = 10

leaving_logs = True


#environment
training_environments = MultipleEnvironments([
                                CartPoleV0() for _ in range(int(training_environments_num) )
                            ])

test_environment = CartPoleV0()

#agent
if model_type == "separate":
    model = SeparateA2CModel(ValueNet(), PolicyNet())
elif model_type == "shared":
    model = SharedA2CModel(DualNet())
else:
    raise Exception("unexpected model_type")

if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

parameters_filename = "parameters/cartpole_agent_parameters.json"

creater = A2CAgentCreater(parameters_filename)
agent = creater.create(model)


#memory
batch_size = agent.batch_size
    
memory_size = int(agent.batch_size/training_environments_num)
shape = (max_sequencies_in_training_memory, memory_size)
training_memories = MultipleSequentialMemories([
                        SequentialMemories(shape) for _ in range(training_environments_num)
                    ])

shape = (episodes_to_test, test_environment.max_steps)
test_memory = SequentialMemories(shape)


#log setting  
if leaving_logs:
    log_folder = "./log/cartpole/" + datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(log_folder, exist_ok=True)

    log_filename =  log_folder + "/test_log.txt"
    logs = [DisplayLog(), FileLog(log_filename)]

    shutil.copy(parameters_filename, log_folder+"/cartpole_agent_parameters.json")
        
    shutil.copy("./cartpole/models.py", log_folder+"/models.py")
        
else:
    logs = [DisplayLog(),]

log = Logs(logs)


#training and test
cumulative_step = 0
State = training_environments.reset()

for epoch in range(0, max_epochs+1):
    #test
    if epoch % test_interval == 0:
        cumulative_rewards = []
        steps = []
        test_memory.clear()
        for episode in range(episodes_to_test):
            state = test_environment.reset()
            cumulative_reward = 0

            for step in range(1, test_environment.max_steps+1):
                action = agent.action(state, deterministic=True)
                (state_dash, reward, done, info) = test_environment.step(action)

                test_memory.append(state, action, reward)
                cumulative_reward += reward

                if done:
                    break

                state = state_dash
            #for step in range(1, test_environment.max_steps+1):

            is_terminal = info["is_terminal"]

            test_memory.end_sequence(state_dash, is_terminal)

            cumulative_rewards.append(cumulative_reward)
            steps.append(step)
        #for episode in range(episodes_to_test):
        
        experiences = test_memory.refer()
        test_loss = agent.estimate(experiences)

        test_stats = {
                        "test_steps":{ "mean":mean(steps) },
                        "cumulative_reward":{ "best":max(cumulative_rewards), "mean":mean(cumulative_rewards), "worst":min(cumulative_rewards) },
                        "test_loss": test_loss
                    }

        training_stats = agent.stats

        record = aggregate_stats(epoch, cumulative_step, test_stats, training_stats)
        log.print(record)

        if leaving_logs:
            filename_string = "Epoch-" + str(epoch).zfill(10)
            model_filename = log_folder +"/" +filename_string + ".model"
            agent.save(model_filename)

    #if step % test_interval == 0:

    for t in range(training_per_epoch):
        training_memories.clear()
        
        #exploration
        #act and get experiences, set of (state, action, reward) and some accompany informations
        while len(training_memories) < batch_size:
            Action_ = agent.action(np.vstack(State))
            Action  = Action_.tolist()

            (State_dash, Reward, Done, Info) = training_environments.step(Action)
            training_memories.append(State, Action, Reward, Done, Info)
        
            State = State_dash

        training_memories.end_sequence(State_dash)

        #training
        experiences = training_memories.refer()
        agent.to(device)
        agent.train(experiences)
        agent.to(torch.device("cpu"))

        cumulative_step += batch_size

    agent.step_scheduler()
#for epoch in range(0, max_epochs+1):

training_environments.close()
test_environment.close()

