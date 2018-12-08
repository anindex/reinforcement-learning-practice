# Week 2 Markov Decision Process - A problem statement of Reinforcement Learning

## Overview of RL

RL has many applications in different fields ranging from Optimal Control, Reward System, Operations Research (Economics), etc. It is indeed a huge distinct branch
in Machine Learning beside Supervised and Unsupervised Learning. There are important aspects in the formulation of RL:

- No supervisor, only reward signal serves as heuristic guidance.
- RL is stochastic, so feedback is not instantaneous.
- The variables are sequential time-dependent.
- Action affects subsequent observable data.

Using RL, we try to build an agent that acts ![alt text](https://latex.codecogs.com/gif.latex?A_t) based on the feedback reward signal ![alt text](https://latex.codecogs.com/gif.latex?R_t) and observation 
![alt text](https://latex.codecogs.com/gif.latex?O_t) each time step t. The environment will receive action ![alt text](https://latex.codecogs.com/gif.latex?A_t) and emit reward ![alt text](https://latex.codecogs.com/gif.latex?R_{t+1}) and observation ![alt text](https://latex.codecogs.com/gif.latex?O_{t+1}) for the next time step:

<p align="center">
  <img src=https://user-images.githubusercontent.com/18066876/49680574-78175b80-fac8-11e8-8c0b-7549ae4b78d8.PNG alt="drawing" width="360" height="360">
</p>

Both agent and environment has its own internal state ![alt text](https://latex.codecogs.com/gif.latex?S_t^a) and ![alt text](https://latex.codecogs.com/gif.latex?S_t^e) respectively. We can formulate agent state as a function of history ![alt text](https://latex.codecogs.com/gif.latex?S_t^a&space;=&space;f(H_t)). 

## Markov Process (Markov Chain)

Markov Process is a mermoryless random process, i.e a sequence of random state ![alt text](https://latex.codecogs.com/gif.latex?S_t) has the property:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P[S_{t&plus;1}|S_t]&space;=&space;P[S_{t&plus;1}|S_1,&space;...,&space;S_t]">
</p>

The property means that the state captures all relevant information from the history, so history can be discarded, assuming the agent is full observability ![alt text](https://latex.codecogs.com/gif.latex?O_t&space;=&space;S_t^a&space;=&space;S_t^e). Intuitively, we can connect from the fact that the environment state is indeed Markov, hence the agent state is also Markov if the system is fully observable.

If we have finite n state ![alt text](https://latex.codecogs.com/gif.latex?S_1,...,S_n), the state transitions P can be characterized by a matrix:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;P_{11}&space;&&space;...&space;&&space;P_{1n}\\&space;...&space;&&space;&&space;\\&space;P_{n1}&space;&&space;...&space;&&space;P_{nn}&space;\end{bmatrix}">
</p>

## Markov Decision Process (MDP)



### MDP Formulation

### Optimal Policy for MDP
