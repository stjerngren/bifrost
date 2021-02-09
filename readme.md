# Bifröst

Bifrost (/ˈbɪvrɒst/) is a bridge between [Apache TVM](https://tvm.apache.org) and [STONNE]().

The name is taken from Norse mythology, where Bifröst is the bridge between Midgard and Asgard

## Dissertation formalities
* [Overleaf Dissertation](https://www.overleaf.com/project/5f756faefef3ec00014e888a)
* [Timelog](https://github.com/axelstjerngren/level-4-project/wiki/Timelog)
* [Meeting Notes](https://github.com/axelstjerngren/level-4-project/wiki/Meeting-Notes)


## Quickstart
You can install and run this project with one command:
```
pip install git+https://github.com/axelstjerngren/level-4-project#"egg=bifrost&subdirectory=bifrost"
```
This will enable to you to use the latest version of Bifröst. 
**N.B** You need to have Apache TVM installed. You can find installation instructions [here](https://tvm.apache.org/docs/install/index.html).

## Build from source

Install Apache TVM using the installation instructions [here](https://tvm.apache.org/docs/install/index.html).

Clone the project and cd into bifrost
```
git clone https://github.com/axelstjerngren/level-4-project
cd level-4-project/bifrost
```
You can now install it by running setup.py:
```
python setup.py install 
```
You can now use Bifrost.

Alternatively, if you are going to make modifications to Bifrost then export it to PYTHONPATH to tell python where to find the library. This way your changes will immeditaly be reflected and there is no need to call setup.py again.
```
export BIFROST=/path/to/level-4-project/bifrost/
export PYTHONPATH=$BIFROST/python:${PYTHONPATH}
```

## Modifying the C++ code 
All of the C++ files can be found in under:
```
level-4-project
|___bifrost
|    |__src
|    |   |__include
|    |   |     |__cost.h
|    |   |
|    |   |__conv_forward.cpp
|    |   |__cost.cpp
|    |   |__json.cpp
|    |   |__etc...
|    |__Makefile
```

Any new .cpp files will be automatically found by the Makefile as long as they are created within the /src folder. Before you compile the code you need STONNE and TVM as enviroment variables (see next section) You can the compile your new code with the following commands:
```
cd bifrost
make -j
```

### C++ depdencies 
To change the C code you need to clone the STONNE and TVM repositories:
```
git clone https://github.com/axelstjerngren/stonne
git clone https://github.com/apache/tvm
```
Keeping these three in the same folder will be useful.
Before you can run **make** you need to export two environment variables:
```
export TVM_ROOT    = path_to_tvm/tvm
export STONNE_ROOT = path_to_stonne/stonne
```
The C++ should now compile correctly when you run **make**.



## Dependecies
Python >=3.8

Remember to add proper credits to JSONCPP: https://github.com/open-source-parsers/jsoncpp



## Run the tests
Bifrost includes a test suite. This will run all implemented layers (conv2d and dense) on STONNE and compare the output against the TVM x86 implementation for correctness. You can run the tests using the following commands:
```
cd bifrost
python setup.py
```
Tested on macOS Big Sur (11.1) and Manjaro 20.2.1 

### Architecture

### Where can I find stuff?
All the agents can be found under project/implementation/agents/. Here you should be able to see:

* base_agent.py - This is the base class all agents extend. The externally exposed API is defined in this file.
* deep_q_learning_agent.py
* deterministic_agent.py
* policy_search_tabular_agent.py
* q_tabular_agent.py
* random_agent.py

It might be a good idea to quickly skim through the base_agent before looking at derivative agents.

### How do I run stuff?
We have taken a "batteries included" type of approach and provided all the different agent we have trained so you don't have to do it yourself. You can simply open up run_eval.ipynb and step through each cell and each agent.

To verify the reproducibility of our code:
All pre-trained agents are stored in folders in the root directory: deep_q, policy_tabular, and q_tabular. By removing all the files in these folders run_eval.ipynb will recognise there are no are pre-trained agents and start training new ones.

If you want to run your own agents this is easy as well. **Even if you're not interested in running anything manually yourself, we recommended that you read through the below to get a deeper understanding of how the different parts of the project interact**

Create a new python file or ipython notebook in the root directory of aiae. All agents are exposed through the project module. You can import them like this:

``` python
from project import DeterministicAgent, DeepQLearningAgent, QTabularAgent, RandomAgent, PolicySearchTabularAgent.
```
Once you have an agent, you can easily initate them and start training. Here is an example of a DeepQLearningAgent:
**All agents use the exact same API of train(), save_policy(), and load_policy(), and run(). The first three methods will obviosuly not do anything when called on a DeterministicAgent or a RandomAgent since they are not reinforcemetn learning agents.**

``` python
deep_q_agent = DeepQLearningAgent() # To run with default hyperparameters
deep_q_agent = DeepQLearningAgent(
        max_steps_per_episode=500,
        discount_factor=0.95,
        epsilon_init=0.01,
        epsilon_decay=0.99995,
        epsilon_min=0.01,
        use_batch_updates=True,
        show=False,
        buffer_size=10000,
        batch_size=128,
        alpha=0.001,
        nn_config=[24, 24]
) # Initialse with some example hyperparameters

# We can now train the agent
deep_q_agent.train(episodes = 50) # This will train on problems list(range(1,10))
deep_q_agent.train(episodes = 50, problems = [2,4,7,9]) # Train on any combinations of specific problems.

# You can also access training statistics:
results = deep_q_agent.results["training"] # Results object explained below

# Please note that you can only train problems sthochasticity and noise set to False. 
# If you for some reason don't agree with this decision, you can access the agent's private _train() function. 
# The train() function wraps the _train() to add functionality such as training on multiple problems and saving 
# statistics about the training data/
# This requires you to intialise a ViRL environment yourself!
# Just because you can doesn't mean you should...

import project.virl 
env = virl.Epidemic(problem_id=0, stochastic=False, noisy=False)
deep_q_agent._train(env, episodes = 50.)

# Once you have trained your agent, you will probably want to test it out!
# Not that if stochastic is true, it will take the average out of 20 runs. 

deep_q_agent.run(
        result_id: str   = "my_trial_run",
        problem_id: int  = 0,
        stochastic: bool = False,
        noise: bool      = False,
  )
  
# Once again you might not agree with some of the default behaviour, then you use
# each agent's private _run() function which will only step through an environment once.
# This requires you to intialise a ViRL environment yourself!

env = virl.Epidemic(problem_id=0, stochastic=False, noisy=False)
deep_q_agent._run(env, result_id: str = "my_trial_run")

# Now you might wonder where the results of the run has gone? Each agent
# has an associated results object where indivuals results from different runs can be accessed
# using a result_id:

results = deep_q_agent.results["my_trial_run"]
results.rewards # All the rewards for a run

results.states# All the visited states in a run. 
#This will not store discretized states even if the agent relies on discretization.

results.actions # All the actions performed during a run

results.get_total_reward() # Get the total reward of a run

# After analysing the data from some runs you might decide that you think the agent performs well. 
# You can then save the policy so that you can reuse it in the future.

# Save to current working directory with the name "model"
deep_q_agent.save_policy()

# Save to current a specified path with a custom name
deep_q_agent.save_policy(path, "name")

# We can also do the opposite and load a policy the same way:

# Save to current working directory with the name "model"
deep_q_agent.load_policy()

# Save to current a specified path with a custom name
deep_q_agent.load_policy(path, "name")


```

### External Sources
* Apache TVM |  | 
* STONNE | | 
* JSONCPP | | 
