There is not any seperate plot function. For testing, just run windy_exp_n_step.py and
enter parameters. For testing programming Question,please enter 1 while choosing step.
For testing bonus, please enter the n you want for n-step sarsa.

under the condition of 1-step sarasa
4-action will lead to 160 - 180 episodes in 8000 steps
8-action will lead to 350 - 380 episodes in 8000 steps
9-action will lead to 280 - 300 episodes in 8000 steps
As a result, 8-action performs better than 4-action and 9 action does better than 4-action
but worse than 8-action

Since we only run 8000 steps, 2-step sarsa performes a little better than
1-step sarsa. When n > 3, 1-step sarsa performs worse than 1-step sarsa.
In this case,2-step sarsa may be the optimal

