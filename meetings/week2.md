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

## Markov Decision Process (MDP)

MDP serves as an effective mathematical foundation for the problem of RL, it formally describes an environment of RL. For instance, almost all RL problems can be formalised as MDPs:

- Optimal Control can be formulated with continuous MDPs
- Partially Observable problems can be converted into MDPs
- Bandits are MDPs with one state. 

MDP fits correctly with the RL characteristics when RL agent has all information about the environment (full observability), where RL agent is stochastic, time-dependent and action affecting subsequent data. This document will introduce MDP definitions and formulations, the next posts will discuss how to solve MDPs and apply it to RL problems.

### Markov Process (Markov Chain)

Markov Process is a mermoryless random process, i.e a sequence of random state ![alt text](https://latex.codecogs.com/gif.latex?S_t) has the property:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P[S_{t&plus;1}|S_t]&space;=&space;P[S_{t&plus;1}|S_1,&space;...,&space;S_t]">
</p>

The property means that the state captures all relevant information from the history, so history can be discarded, assuming the agent is full observability ![alt text](https://latex.codecogs.com/gif.latex?O_t&space;=&space;S_t^a&space;=&space;S_t^e). Intuitively, we can connect from the fact that the environment state is indeed Markov, hence the agent state is also Markov if the system is fully observable.

If we have finite n state ![alt text](https://latex.codecogs.com/gif.latex?S_1,...,S_n), the state transitions P can be characterized by a matrix:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;P_{11}&space;&&space;...&space;&&space;P_{1n}\\&space;...&space;&&space;&&space;\\&space;P_{n1}&space;&&space;...&space;&&space;P_{nn}&space;\end{bmatrix}">
</p>

### Markov Reward Process (MRP)
Basically, MRP is Markov chain with values for each state. The value of each state measures how good to be in that state. To be concrete, MRP is a tuple ![alt text](https://latex.codecogs.com/gif.latex?(S,&space;P,&space;R,&space;\gamma)):

- S is a finite set of states
- P is state transition probability matrix, ![alt text](https://latex.codecogs.com/gif.latex?P_{ss'}&space;=&space;P[S_{t&plus;1}=s'|S_t=s])
- R is a (immediate) reward function for each state, ![alt text](https://latex.codecogs.com/gif.latex?R_s=E[R_{t&plus;1}|S_t=s])
- ![alt text](https://latex.codecogs.com/gif.latex?\gamma) is a discount factor ranging from 0 to 1.

The value function v(s) computes value for each state is the expected reward return starting from state s:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v(s)&space;=&space;E[G_t|S_t=s]">
</p>

where return G is

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?G_t&space;=&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;...=\sum_{k=0}^\infty&space;\gamma&space;^kR_{t&plus;k&plus;1}">
</p>

Intuitively, the function value averages all possible reward return path from state s to the terminal state of MRP (terminal state to avoid infinite return), where each path has a unique return G. The factor ![alt text](https://latex.codecogs.com/gif.latex?\gamma) discounts G to also avoid infinite return G when there are infinite state sequences, and to represent uncertainty of far future reward and favoring immediate rewards.

The value function can be in the form of Bellman equation:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v(s)=E[R_{t&plus;1}&plus;\gamma&space;v(S_{t&plus;1})|S_t=s]">
</p>

 or
 
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v(s)=R_s&plus;\gamma&space;\sum_{s'\in&space;S}P_{ss'}v(s')">
</p>

To be able to solve for v(s), we must represent Bellman function in matrix form:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v&space;=&space;R&space;&plus;&space;\gamma&space;Pv">
</p>

and solve for v:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v&space;=&space;(I&space;-&space;\gamma&space;P)^{-1}R">
</p>

### Markov Decision Process (MDP)

MDP is a Markov reward process with decisions to act on the environment, and then the enviroment will stochastically transition to the next state s'. Please note that all states in MDP are Markov. A MDP is a tuple ![alt text](https://latex.codecogs.com/gif.latex?(S,&space;A,&space;P,&space;R,&space;\gamma))

- S is a finite set of states
- A is a finite set of actions
- P is state transition probability matrix, ![alt text](https://latex.codecogs.com/gif.latex?P_{ss'}^a&space;=&space;P[S_{t&plus;1}=s'|S_t=s,&space;A_t=a])
- R is a (immediate) reward function for each state, ![alt text](https://latex.codecogs.com/gif.latex?R_s^a=E[R_{t&plus;1}|S_t=s,A_t=a])
- ![alt text](https://latex.codecogs.com/gif.latex?\gamma) is a discount factor ranging from 0 to 1.

To extend from MRP, MDP defines a policy distribution over a set of agent's actions on state s:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\pi&space;(a|s)=P[A_t=a|S_t=s]">
</p>

The policies are depend on the current state s (not the history) and are stationary(time-independent).
MDP also redefines the state-value function and action-value function, which are the expected return starting from state s and then following the policy ![alt text](https://latex.codecogs.com/gif.latex?\pi):

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_\pi&space;(s)=E_\pi[G_t|S_t=s]">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_\pi(s,a)&space;=&space;E_\pi[G_t|S_t=s,A_t=a]">
</p>

we then rewrite these equation as Bellman Expectation equations:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;E_\pi[R_{t&plus;1}&plus;\gamma&space;v_\pi(S_{t&plus;1})|S_t=s]">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_\pi(s,a)&space;=&space;E_\pi[R_{t&plus;1}&plus;\gamma&space;q_\pi(S_{t&plus;1},A_{t&plus;1})|S_t=s,A_t=a]">
</p>

After simple transformations, we can see their inter-relation as follow:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;\sum&space;_{a&space;\in&space;A}\pi(a|s)q_\pi(s,a)">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_\pi(s,a)&space;=&space;R_s^a&plus;\gamma\sum_{s'&space;\in&space;S}P_{ss'}^av_\pi(s')">
</p>

which leads us to the state-value transition function and action-value transition function:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=\sum_{a&space;\in&space;A}\pi(a|s)(R_s^a&plus;\gamma\sum&space;_{s'&space;\in&space;S}P_{ss'}^av_\pi(s'))">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_\pi(s,a)&space;=&space;R_s^a&plus;\gamma\sum_{s'&space;\in&space;S}P_{ss'}^a\sum&space;_{a'&space;\in&space;A}\pi(a'|s')q_\pi(s',a')">
</p>

We can solve MDP state-value function by Bellman Expectation equation in matrix form:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_\pi&space;=&space;(I&space;-&space;\gamma&space;P^\pi)^{-1}R^\pi">
</p>

where 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P_{ss'}^\pi&space;=&space;\sum&space;_{a&space;\in&space;A}\pi(a|s)P_{ss'}^a">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?R_{s}^\pi&space;=&space;\sum&space;_{a&space;\in&space;A}\pi(a|s)R_{s}^a">
</p>

It is computationally hard to compute state-value function, which is O(n^3) for n states. We must use iterative methods for large set of states, e.g Dynamic Programming, Monte-Carlo evaluation, Temporal-Different learning, etc. 


### Optimal Policy for MDP

We define the optimal state-value function and action-value function are the maximum of these function over all policies:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_*(s)&space;=&space;\underset{\pi}{max}&space;v_\pi(s)">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_*(s,a)&space;=&space;\underset{\pi}{max}&space;q_\pi(s,a)">
</p>

The optimal value function specifies the best possible performance in MDP. We reply on MDP theorem to find optimal policies:

- There exists an optimal policy that is better than all other policies ![alt text](https://latex.codecogs.com/gif.latex?\pi_*&space;\geq&space;\pi,&space;\forall&space;\pi)
- There may be multiple solution to achieve optimal state-value function and action-value function.

If we know optimal ![alt text](https://latex.codecogs.com/gif.latex?q_*(s,a)), we immediately have the optimal policy. We can solve recursively for optimal state-value function and action-value function through these Bellman optimality function:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?v_*(s)&space;=&space;\underset{a}{max}R_s^a&plus;\gamma\sum&space;_{s'&space;\in&space;S}P_{ss'}^av_*(s')">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?q_*(s,a)&space;=&space;R_s^a&plus;\gamma\sum&space;_{s'&space;\in&space;S}P_{ss'}^a\underset{a'}{max}q_*(s',a')">
</p>
Again, Bellman optimality equations is non-linear and have no close form solution. We must use iterative methods to solve them, e.g Value Iterative, Policy iteration, Q-learning, Sarsa.
