# Week 1 Blog of An Outsider's Tour of Reinforcement Learning
## Introduction
This document summarizes what I have learnt about the intersection of Control Theory and Reinforcement Learning (RL) topic. The topic was greatly introduced by Ben Recht, which he proposed this new exciting field the name **Actionable Intelligence**.

To begin, this is the introductory headline of the field: 

> Actionable Intelligence is the study of  how to use past data to enhance the future manipulation of a dynamical system

I know some Computer Science guys at my Uni said that Deep RL is the key to achieve The Singularity (lol), and yes there are strong supports that RL has defeated human in Go, Atari game, etc. To be honest, I doubted about that audacious blame. At that time, I thought that Deep RL achieved these performances in perfect environment without measurement error and no hidden game states, but how can we so sure that it will achieve the same in dynamic and imperfect environment of industrial systems? 

The tour of Ben Recht indeed clears my doubts, there is a whole pool of problems to solve. Why is it so hard to optimal control a system given unknown dynamic states? RL has to be the one that could easily solve this kind of problem. The answer is simple, the stakes could not be higher if we would like to guarantee safety executions and valuable gains (both are defined by reward function) by a series of action in uncertainty. Afterall, RL is to find (randomly or heuristic) a series of action **u** to maximize reward function **r**.

## A test-beds for everyone to test: Linear Quadratic Regulator (LQR)

As Prof. Recht stated his point:

> If a machine learning does crazy things  when restricted to  linear models, it's going to do crazy things on complex nonlinear models too.

And reversely, if your ML algorithm has good results on non-linear models, it has to be get acceptable results on linear models too. Otherwise, it is unreasonable to trust your model. Therefore, LQR has come to save the day as a test-bed for everything new pop out of your head:  



## Approaches to solve LQR with RL


## Questions?

Indeed, I write this blog with only high-level intuition with high error (or variance of belief) :). There are some questions that bug me while reading this topic due to my misunderstanding or my imcompetency.

**Model-based methods**

- After model estimation, we plug the model f as neural network to state transitions and treat it as true function, then how we can estimate its uncertainty like we did in linear model?

**Model-free methods** 
- Does ADP treat both cost function and state-transition function into one big cost function and use data from oracle and measurement to gradually estimate the cost function to minimize it? 

- How do model-free methods of RL acquire system states ![alt text](https://latex.codecogs.com/gif.latex?x_t) value? By mean of sampling online or offline? 

**Approximate Dynamic Programming**
- (Too hard to understand what is happening?) We must know the terminal condition to estimate Q and then optimize action u?
Again, does ADP sample (or measure) unknown series of state ![alt text](https://latex.codecogs.com/gif.latex?x_t)? 

**Direct Policy Search**
- Is it some form of sampling data z over a parametrized distribution and gradually improving shape of distribution to minimize cost function (via the distribution)? 
- Need clarification in the Policy Gradient slide. How the distribution of u is updated?

**Misc**
- Using bootstrap in simulation to estimate the error of model?



