Part 1,2:
	In part1, the agent is agent_hw6.py. The run this experiment, just run exp_hw6.py and the use the plot.py to plot the learning curve.

Part 3:
	Run part3.py to generate data and then user plot_p3,py to plot the cost-to-go graph. I saved the data in both npy and file types.

Bonus part:
	A new expirement called bonus.py and a new agent called bonus_agent.py. In this case, I indicate the mean and standard error in both graph and the program. With 50 runs of 200 episodes, what I changed is the learning rate alpha from 0.1/numtilings to 0.5/numtilings. It helped a lot to the performance. The program converges much faster than the original setting.
	The standard error of 2 curves are all around 110, the mean of the one with larger alpha is -27348.08 which is much lower than -39958.68 of the original setting.