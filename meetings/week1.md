# Week 1 Blog of An Outsider's Tour of Reinforcement Learning - Ben Recht
## Introduction
This document summarizes what I have learnt about the intersection of Control Theory and Reinforcement Learning (RL) topic. The topic was greatly introduced by Ben Recht, which he proposed this new exciting field the name **Actionable Intelligence**.

To begin, this is the introductory headline of the field: 

> Actionable Intelligence is the study of  how to use past data to enhance the future manipulation of a dynamical system

I know some Computer Science guys at my Uni said that Deep RL is the key to achieve The Singularity (lol), and yes there are strong supports that RL has defeated human in Go, Atari game, etc. To be honest, I doubted about that audacious blame. At that time, I thought that Deep RL achieved these performances in perfect environment without measurement error and no hidden game states, but how can we so sure that it will achieve the same in dynamic and imperfect environment of industrial systems? 

The tour of Ben Recht indeed clears my doubts, there is a whole pool of problems to solve. Why is it so hard to optimal control a system given unknown dynamic states? RL has to be the one that could easily solve this kind of problem. The answer is simple, the stakes could not be higher if we would like to guarantee safety executions and valuable gains (both are defined by reward function) by a series of action in uncertainty. Afterall, RL is to find (randomly or heuristic) a series of action **u** to maximize reward function **r**.

## A test-bed for everyone to test: Linear Quadratic Regulator (LQR)

As Prof. Recht stated his point:

> If a machine learning does crazy things  when restricted to  linear models, it's going to do crazy things on complex nonlinear models too.

And reversely, if your ML algorithm has good results on non-linear models, it has to be get acceptable results on linear models too. Otherwise, it is unreasonable to trust your model. Therefore, LQR has come to save the day as a test-bed for everything new popped out of your head. 

LQR is a special case of optimal control problem when we know the state transitions function and we design the cost function to represent our task:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?min&space;E_e[\sum_{t=1}^{T}C_t(x_t,&space;u_t)]">
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;f_t(x_t,&space;u_t,&space;e_t)">
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?u_t&space;=&space;\pi_t(\tau_t)">
</p>

With LQR (continuous version), the state transition function is linear and the cost becomes quadratic:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?E[\frac{1}{T}\sum_{t=1}^{T}x_t^TQx_t&plus;u_t^TRu_t&plus;x_T^TP_Tx_T]">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;Ax_t&space;&plus;&space;Bu_t&space;&plus;&space;e_t">
</p>

Intuitively, you must find a series of action ![alt text](https://latex.codecogs.com/gif.latex?u_t) in order to minimize the quadratic cost expected over error (to get mean error). Next section will introduce some approaches to find the optimal (or not) policy and how fast they converge to minimum cost with a fixed amount of samples. The raising question is how we can balance between optimal cost and computation to process as least as possible many samples. 

I have not implemented any approaches in RL to try to solve LQR yet, but it seems pretty straightforward to implement. However, Prof. Recht mentioned LQR is not easy, "magic tends to happen in LQR". I do not know if this is true until I get my hand dirty on evaluating these approaches.

## Approaches to solve LQR with RL

In the above LQR problem, we design the cost function based on our demands and if we fully know the state transitions functions, we can solve LQR directly using Batch Optimization or Dynamic Programming. And the solution is described as below.

For finite-time horizon, discrete-time LQR:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?u_t&space;=&space;-K_tx_t">
</p>

where 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?K_t&space;=&space;-(B^TP_{t&plus;1}B&plus;R)^{-1}B^TP_{t&plus;1}A">
</p>

and 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P_t&space;=&space;Q&space;&plus;&space;A^TP_{t&plus;1}A&space;-&space;A^TP_{t&plus;1}B(R&space;&plus;&space;B^TP_{t&plus;1}B)^{-1}B^TP_{t&plus;1}A">
</p>

When we know the state transition, we know A and B in linear equation. The optimal solution is founded by recursively find ![alt text](https://latex.codecogs.com/gif.latex?P_{t+1}) from terminal condition ![alt text](https://latex.codecogs.com/gif.latex?P_T). Then we can find series ![alt text](https://latex.codecogs.com/gif.latex?u_t). 

When the time T approaches infinity, the optimal solution of **u** becomes static. Solving Riccati equation of P, we get:

<p align="center">
  <img src=https://latex.codecogs.com/gif.latex?P&space;=&space;Q&space;&plus;&space;A^TPA&space;-&space;A^TPB(R&space;&plus;&space;B^TPB)^{-1}B^TPA>
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?K&space;=&space;-(B^TPB&plus;R)^{-1}B^TPA">
</p>

As we can see, the optimal solution **u** is independence of error variance, which means more noise do not affect system control. In infinite-time scheme, solution **u** is time-invariant.

The questions is, when we do not know state transition function, we do not know A, B in advanced, how can we solve LQR for optimal control? This is the place that RL shined. 

We have these roads to follow:

1. Identify everything (the most masochistic way lol) about system dynamic using physics and finite element methods.
2. Identify a coarse model for the system dynamic
3. We will be blunt :) and map directly data from sensor models to output control (e.g neural network)

### Model-based approaches

I am personally like the first approach, because that's what Control Theory guys usually do anyway :). They called this approach System Identification. However, in ML manner, we do not use the glory of Physics to model the system. What we are going to do is trying to learn the state-transition function, which means we will fit the matrix A and B with some cost function by choices. For example, we use L2 loss to fit A, B:

<p align="center">
  <img src=https://latex.codecogs.com/gif.latex?\hat{A},&space;\hat{B}&space;=&space;\underset{A,&space;B}{min}&space;\sum_{t=1}^{T-1}||x_{t&plus;1}&space;-&space;Ax_t&space;-&space;Bu_t||^2>
</p>

Then we plug these matrices to state transition function with an error ![alt text](https://latex.codecogs.com/gif.latex?v_t):

<p align="center">
  <img src=https://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;\hat{A}x_t&plus;\hat{B}u_t&space;&plus;&space;v_t>
</p>

Actually, if we treat ![alt text](https://latex.codecogs.com/gif.latex?\hat{A},&space;\hat{B}) as true dynamic matrices, it does not feel right. So far, this approach finds solution **u** based on what we just estimated ![alt text](https://latex.codecogs.com/gif.latex?\hat{A},&space;\hat{B}). Here is the burning question: How can we model the uncertainty (errors) of our estimation and then use that to design solution **u** to robust control the system?

To answer that question, there are recent works of (again) Prof.Recht's group about Coarsed-ID control addressing this problem. These works will be discussed in next posts.

### Model-free approaches

This section is huge! I cannot write the section down with few lines, it is even fit to have many intensive courses in RL. And yes, this is the realm of RL, everyone like it :). This section serves as introductory point for the next few posts. Basically, as the name stated, we do not care about the system dynamic, what we do is to mitigate cost function by:

- Iteratively refining cost function based on measured system state and estimating policy ![alt text](https://latex.codecogs.com/gif.latex?u_t) from the cost function each time step: **Approximate Dynamic Programming**.
- Defining a parametrized distribution ![alt text](https://latex.codecogs.com/gif.latex?p(u;\vartheta)) over ![alt text](https://latex.codecogs.com/gif.latex?u_t) and randomly searching ![alt text](https://latex.codecogs.com/gif.latex?u_t) to refine iteratively the parametrized distribution ![alt text](https://latex.codecogs.com/gif.latex?p(u;\vartheta)). The purpose is to minimize cost function expected over that distribution: **Direct Policy Search**.  

For now, we do not go to detail of these methods. PID is also a model-free method that find **u** based on just the error, but PID can  learn to compensate for poor models, changing conditions? This leads us to a burning topic from PID to RL: 

> How well must we understand the system in order to optimal control it? 

The next few posts we will discuss how we balance between optimal policy solution versus computation cost to process the system state samples and how fast these methods find solution that leads system to stability.


## Questions?

Indeed, I write this blog with only high-level intuition with high error (or variance of belief) :). There are some questions that bug me while reading this topic due to my misunderstanding or my imcompetency.

**Model-based methods**

- After model estimation, we plug the model f as neural network to state transitions and treat it as true function, then how we can estimate its uncertainty like we did in linear model?

**Model-free methods** 
- Does ADP treat both cost function and state-transition function into one big cost function and use data from measurement (like measure state x) to gradually estimate the cost function to minimize it? 

- How do model-free methods of RL acquire system states ![alt text](https://latex.codecogs.com/gif.latex?x_t) value? By mean of measurement online or offline? 

**Approximate Dynamic Programming**
- (Too hard to understand what is happening?) We must know the terminal condition to estimate Q and then optimize action u?
Again, how does ADP sample (or measure) unknown series of state ![alt text](https://latex.codecogs.com/gif.latex?x_t)? 

**Direct Policy Search**
- Is it some form of sampling data z over a parametrized distribution and gradually improving shape of distribution to minimize cost function (via the distribution)? If so, like Prof. Recht said, this paradigm is just like random search over action **u** space to minimize cost?

- Need clarification in the Policy Gradient slide. How the distribution of u is updated?

**Misc**
- Using bootstrap in simulation to estimate the error of model?


## References

- [An Outsider's Tour of Reinforcement Learning by Ben Recht](http://www.argmin.net/2018/06/25/outsider-rl/)
- [A Tour of Reinforcement Learning: The View from Continuous Control](https://arxiv.org/abs/1806.09460)
- [Optimization Perspectives on Learning to Control ICML 2018 Tutorial](https://people.eecs.berkeley.edu/~brecht/l2c-icml2018/)
