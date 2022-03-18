#!/usr/bin/env python
# coding: utf-8

# # RBC with two states using all-in-one expectation operator by Maliar, Maliar and Winant (2021) "Deep learning solution method with all-in-one expectation operator"
# -  Capital is the endogeneous state variable and productivity is exogenous
# - This notebook is a simplified version of the companion notebook by Maliar, Maliar and Winant (2019) in the paper "Will Artificial Intelligence Replace Computational Economists Any Time Soon?", CEPR working paper DP 14024, (https://lmaliar.ws.gc.cuny.edu/files/2019/10/CEPR-DP14024.pdf). The original code was for a consumption-saving problem with five exogenous states with a borrowing constraint. Texts and codes remain almost the same from the original one. 

# # Deep learning solution method with all-in-one expectation operator
# 
# - This companion notebook presents one of the deep learning methods for solving dynamic economic models introduced by Maliar, Maliar and Winant (2019) in the paper "Will Artificial Intelligence Replace Computational Economists Any Time Soon?", CEPR working paper DP 14024, (https://lmaliar.ws.gc.cuny.edu/files/2019/10/CEPR-DP14024.pdf).
# - In the paper, we offer a unified approach for casting three fundamental objects of economic dynamics -- lifetime reward, Bellman equation and Euler equation -- into objective functions of the deep learning framework. 
# - In this notebook, we illustrate only one of the three approaches -- the Euler residual minimization. 
# - We solve a cannonical consumption-saving problem with occasionally binding borrowing constraint and four exogenous stochastic shocks. 
# - We parameterize the agent's decision function with a multilayer neural network, and we perform training using stochastic optimization, namely, in each iteration, we train the model on just one or few grid points that are randomly drawn from the state space (instead of conventional fixed grid with a potentially large number of grid points constructed by a discretization of the state-space). 
# - Our objective function -- the sum of squared residuals in the Euler equation -- has two types of expectation operators, one is with respect to current state variables (which arises because grid points that are randomly drawn from the state space), and the other is with respect to future state variables (which arises because next-period shocks are randomly drawn from the given distributions).
# - We construct an all-in-one expectation method that merges the two expectation operators into one. Namely, we use two independent random draws for evaluating two terms of a squared residual -- this method eliminates the correlation between the terms and pulls the expectation operator out of the square. Our all-in-one expectation operator allows for efficient parallel calculations and reduces greately the cost of training deep neural networks.
# - In our 5-dimensional problem, we obtain a solution in a mintue or so. Importantly, our DL method is highly scalable and can solve very large models intractable with other methods. In particular, in the paper, we solve Krusell and Smith's (1998) model with 1,000 of heterogeneous agents by constructing decision function of 2,001 state variables.

# In[1]:

import tensorflow as tf
import numpy as np
from math import sqrt
# from matplotlib import pyplot as plt
# from tqdm import tqdm as tqdm         # tqdm is a nice library to visualize ongoing loops
import datetime
# followint lines are used for indicative typing
from typing import Tuple
class Vector: pass


# ## The model
# 
# We consider the following consumption-saving problem: 
# 
# $$
# \begin{gather*}
# \underset{\left\{ c_{t},k_{t+1}\right\}_{t=0}^{\infty }}{\max }E_{0}\left[
# \sum_{t=0}^{\infty }\exp (\delta_{t})\beta ^{t}u\left( {c_{t}}\right)\right]  \\
# \text{s.t. }k_{t+1}= \exp(y_t) k^{\alpha}_{t} + (1-\delta) k_{t} -c_{t}, \\
# c_{t}\leq w_{t} = \exp(y_t) k^{\alpha}_{t} + (1-\delta) k_{t},
# \end{gather*}
# $$
# 
# 
# where $c_{t}$ is consumption; $w_{t}$ is the beginning-of-period
# resource; $\beta \in \left[ 0,1\right)$ is a subjective discount
# factor; and initial condition $\left( z,w\right)$ is given.
# There is an occasionally binding inequality constraint: consumption $c_{t}$
# cannot exceed $w_{t}$. There is one different exogenous
# state variable, productivity $y_{t}$
# The productivity is $z_{t}=\exp(y_{t})$. All
# exogenous variables  follow AR(1) processes: 
# 
# 
# $$
# \begin{eqnarray*}
# {y_{t+1}} &=&\rho_{y}{y_{t}}+\sigma_{y}\epsilon^{y}_{t+1}, \\
# \end{eqnarray*}
# $$
# 
# 
# where $\epsilon^i_t \sim \mathcal{N}\left( 0,1\right)$. We assume the Cobb-Douglas utility function $u\left( {c_{t}}\right) =\frac{1}{1-\gamma }\left( c_{t}^{1-\gamma }-1\right)$.

# In[3]:


# Model parameters
α = 0.5 #production function parameter
β = 0.9 #discount factor
γ = 2.0 #risk aversion
δ = 0.15 #depreciation rate
ρ_y = 0.95 #AR process for productivity
σ_y = 0.01
A = 0.522224

# A = 1


# ## Stochastic solution domain
# 
# Our DL framework converts a dynamic stochastic model into a static optimization problem. We will solve the static problem on a random grid of points which are drawn from the following 2-dimentional domain:
# 
# - $y$ is drawn from its ergodic distribution (recall that for an AR(1) process $x$ with autocorrelation $\rho$ and conditional standard deviation $\sigma$, the ergodic distribution is normal with zero mean and standard deviation $\sigma_x= \frac{\sigma}{\sqrt{1-\rho^2}}$).
# - $k$ is drawn from a uniform distribution within an interval: $k\in[k_{\min}, k_{\max}]$.

# In[4]:


# Standard deviations for ergodic distributions of exogenous state variables

σ_e_y = σ_y/(1-ρ_y**2)**0.5


# bounds for endogenous state variable
kmin = 0.9
kmax = 2.1


# In the recursive form, the solution can be characterized by the Kuhn-Tucker
# (KT) conditions 
# 
# $$
# \begin{equation*}
# a\geq 0,\quad b\geq 0\quad and\quad ab=0,
# \end{equation*}
# $$
# 
# where $a$ is the share of wealth that goes to savings and $b$ is the
# Lagrange multiplier
# 
# $$
# \begin{eqnarray*}
# a &\equiv &w-c, \\
# b &\equiv &u^{\prime }(c)-\beta \overline{r}E_{\epsilon }\left[ u^{\prime
# }\left( c^{\prime }\right) \exp \left( \delta ^{\prime }-\delta +r^{\prime
# }\right) \right] .
# \end{eqnarray*}
# $$
# 
# (In the absence of borrowing constraint $b=0$, the KT conditions lead to the familiar
# Euler equation). 
# 
# Inequality constraints are not directly compatible with the deep learning
# framework developed in the paper, so we reformulate the KT conditions as a set of equations that hold with equality. We use a smooth representation of the KT conditions, called
# the Fischer-Burmeister (FB) function, which is differentiable 
# 
# $$
# \begin{equation*}
# \Psi_{FB}\left( a,b\right) =a+b-\sqrt{a^{2}+b^{2}}=0.
# \end{equation*}
# $$
# 
# The restriction $\Psi_{FB}\left( a,b\right) =0$ is also equivalent to the
# KT conditions. 
# 
# For numerical treatment, we rewrite the $\Psi_{FB}$ function in the following
# unit-free form 
# 
# $$
# \begin{equation*}
# \Psi_{FB}\left( 1-\zeta ,1-h\right) =(1-\zeta)+(1-h)-\sqrt{(1-\zeta)^{2}+(1-h)^{2}}=0,
# \end{equation*}
# $$
# 
# where $\zeta $ and $h$ are respectively the consumption share and normalized
# Lagrange multiplier 
# 
# $$
# \begin{eqnarray*}
# \zeta  &\equiv &\frac{c}{w}, \\
# h &\equiv &\beta \overline{r}E_{\epsilon }\left[ \frac{u^{\prime }\left(
# c^{\prime }\right) }{u^{\prime }(c)}\exp \left( \delta ^{\prime }-\delta
# +r^{\prime }\right) \right] .
# \end{eqnarray*}
# $$
# 
# In particular, $\zeta $ belongs to the interval $\left[ 0,1\right] $ which
# is a convenient domain for defining neural network. In turn, $h$ is always positive and is normalized 
# to be around one: we will parameterize it with neural network in the way 
# that ensures that it is nonnegative.

# In[5]:


# Here is the  Fischer-Burmeister (FB) in TensorFlow:
min_FB = lambda a,b: a+b-tf.sqrt(a**2+b**2)


# ## Parameterizing decision functions with neural network
# 
# There are many different decision functions that we can approximate for characterizing the solution, including
# consumption, next-period income, etc. We chose to approximate the two functions that
# we defined earlier: the share of consumption, $\zeta \equiv \frac{c}{w}$, and the
# normalized Lagrange multiplier $h$.
# Since the model is stationary, 
# we look for a decision rule 
# 
# $$
# \begin{equation*}
# \left( 
# \begin{matrix}
# \zeta  \\ 
# h%
# \end{matrix}%
# \right) =\varphi (s;\theta ),
# \end{equation*}
# $$
# 
# where $s=(r, \delta, q, p, w)$ is the 5-dimensional state space, and $\varphi$ is a function to be determined.
# 
# A common approach in computational economics is to approximate an unknown function $\varphi$ using some flexible function family $\varphi(...;\theta)$ parameterized by a vector of coefficients $\theta$, e.g., a polynomial family. Neural networks are just a special family of approximating functions. A distinctive feature of neural networks is that they have a nonlinear dependence of the approximation function on the coefficients $\theta$. TensorFlow contains a submodule keras, which makes it easy to build such a network. Below, we build the multilayer perceptrion: a 2 hidden layers 32x32x32x2 network with relu activation functions and linear outputs

# In[6]:


# construction of neural network
layers = [
    tf.keras.layers.Dense(32, activation='relu', input_dim=2, bias_initializer='he_uniform'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)
]
perceptron = tf.keras.Sequential(layers)


# Next, we create the decision rule which takes as input 2 vectors of the same size $n$ for the states $y$, $k$ and returns two vectors of size $n$ for $\zeta$ and $h$, respectively. We use different nonlinear transformations for the two decision functions, namely, sigmoid/logistic and exponential ones:
# 
# 
# $$
# \begin{equation*}
# \varphi(s;\theta)=\left(\begin{matrix}\frac{1}{1+e^{-\mathcal{N}_1(s;\theta)}}\\ 
# \exp(\mathcal{N}_2(s;\theta))\end{matrix}\right)
# \end{equation*}
# $$
# 
# 
# where $\mathcal{N}_1$ and $\mathcal{N}_2$ denote first and second component of neural network output. This transformation ensures $\zeta\in[0,1]$ and $h>0$

# In[7]:


def dr(y: Vector, k: Vector)-> Tuple[Vector, Vector]:

    # we normalize exogenous state variables by their 2 standard deviations 
    # so that they are typically between -1 and 1 

    # y = y/σ_e_y/2
    
    # we normalze capital to be between -1 and 1
    # k = (k-kmin)/(kmax-kmin)*2.0 

    # we prepare input to the perceptron
    s = tf.concat([_e[:,None] for _e in [y,k]], axis=1) # equivalent to np.column_stack

    x = perceptron(s) # n x 2 matrix 
   
    # consumption share is always in [0,1]
    ζ = tf.sigmoid( x[:,0] )
    
    # expectation of marginal consumption is always positive
    h = tf.exp( x[:,1] )
    
    return (ζ, h)


# Finally, as an illustration, we plot the initial guess of decision rules against $w$. Note that the coefficients of the perceptron are initialized with random values, so that each run will provide a different plot. Here, we are using TensorFlow in an eager mode, i.e., calculations are returned immediately, so that the library essentially behaves in the same way as numpy, and is in fact mostly compatible with it.

# In[8]:


kvec = np.linspace(kmin, kmax, 100)
ζvec, hvec = dr(kvec*0, kvec)


# # In[9]:


# # plt.plot(kvec, kvec, linestyle='--', color='black')
# plt.plot(kvec, ζvec*kvec)
# plt.xlabel("$k_t$")
# plt.ylabel("$c_t$")
# plt.title("Initial Guess")
# plt.grid()


# ## Residuals in the model's equations 
# 
# To identify the unknown decision functions for $\zeta$ and $h$, we use two modelp's equations,
# namely, the definition of normalized Lagrange multiplier and the FB function representing the KT conditions, respectively:
# 
# $$
# \begin{gather*}
# h=\beta \overline{r}E_{\epsilon }\left[ \frac{u^{\prime }\left(
# c^{\prime }\right) }{u^{\prime }(c)}\exp \left(   \delta ^{\prime }-\delta
# +r^{\prime }\right)  \right] , \\
# \Psi_{FB}\left( 1-\zeta ,1-h\right) =0
# \end{gather*}
# $$
# 
# where $\epsilon'=(\epsilon'_r,\epsilon'_\delta,\epsilon'_q,\epsilon'_p)$.
# 
# We do not need to include the definition $\zeta = \frac{c}{w}$ because we will impose it to hold exactly in the solution by setting $c=w\zeta $ and $c^{\prime }=w^{\prime }\zeta ^{\prime }$.
# 
# We next construct the residuals in the above two equations which we will minimize. For given vectors of current state $s=(r,\delta ,q,p,w)$ we use next-period shocks $\epsilon'$ to compute realized residuals:
# 
# $$
# \begin{equation*}
# \begin{matrix}
# R_1(s,\epsilon')=\beta \overline{r} \frac{u^{\prime }\left(
# c^{\prime }\right) }{u^{\prime }(c)}\exp \left( \delta ^{\prime }-\delta
# +r^{\prime }\right) -h,
# \\ 
# R_2(s)=\Psi_{FB}\left( 1-\zeta ,1-h\right), 
# \end{matrix}
# \end{equation*}
# $$
# 
# where the transition equation is $w^{\prime }=\left( w-c\right) \overline{r%
# }\exp (r)+\exp (y)$.

# In[10]:


def Residuals(e_y: Vector, y: Vector, k: Vector):

    # all inputs are expected to have the same size n
    n = tf.size(y)

    # arguments correspond to the values of the states today
    ζ, h = dr(y, k)
    # transitions of the exogenous processes
    ynext = y*ρ_y + e_y
    
    # transition of endogenous states (next denotes variables at t+1)
    c = ζ * k
    knext = tf.exp(y) * A * tf.pow(k,α) + (1-δ) * k - c


    ζnext, hnext = dr(ynext, knext)
    cnext = ζnext * knext
    #Euler Residual
    rnext = 1-δ + α * tf.exp(ynext) * A* tf.pow(knext, α-1)
    R1 = β*(cnext/c)**(-γ)*rnext - h
    R2 = min_FB(1-h,1-ζ)
    
    return (R1,R2)


# ## The expected squared sum of residuals
# 
# We construct the objective function for minimization as the squared sum of
# two residuals in the two model's equations on a given 5-dimensional domain $s=(r,\delta ,q,p,w)$: 
# 
# $$
# \begin{equation*}
# \Xi (\theta )=E_{s}\left[ \left( E_{\epsilon }\left[ R_1(s,\epsilon')\ \right] \right) ^{2}+v\left( R_2(s)\right) ^{2}\right] ,
# \end{equation*}
# $$
# 
# where $v$ is the exogenous relative weights of the two residuals in the
# objective function. We placed the first residual $R_1(s,\epsilon')$ under the expectation
# operator $E_{\epsilon }$ across next-period shocks $\epsilon' =\left(
# \epsilon' _{r},\epsilon' _{\delta },\epsilon' _{q},\epsilon' _{p}\right) $ as is
# required by the definition of $h$; the second residual $R_2(s)$ does not include random variables  
# and requires no expectation operator. The value of the objective function $\Xi (\theta )$
# depends on the coefficients $\theta $ because these coefficients determine the choices via  
# 
# $$
# \begin{equation*}
# \left( 
# \begin{matrix}
# \zeta  \\ 
# h%
# \end{matrix}%
# \right) =\varphi (s;\theta ).
# \end{equation*}
# $$
# 
# A shortcoming of the constructed objective function is that it requires
# a potentially costly evaluation of two nested expectation operators: for
# each random grid point $s=(r,\delta ,q,p,w)$, we need to construct a
# separate approximation of the expectation function $E_{\epsilon }\left[
#  R_1(s,\epsilon')\right] $ by considering a potentially large
# number of next period shocks $\epsilon' =\left( \epsilon' _{r},\epsilon'
# _{\delta },\epsilon' _{q},\epsilon' _{p}\right) $. In particular, if there are 
# $n$ grid points and $J$ next-period shocks, we have $n\times J$ function
# evaluations.

# ## All-in-one expectation function
# 
# In the paper, we introduce a technique which we call an all-in-one expectation operator
# that makes it possible to merge the two expectation operators into a single
# one. This technique relies on a simple result from probability theory that
# says that for two random variables $a$ and $b$, which are independent and
# follow the same distribution, we have $E[a]^{2}=E[a]E[b]=E[ab]$. 
# 
# Therefore, we replace $\left( E_{\epsilon }\left[  R_1(s,\epsilon')\right] \right) ^{2}$ by the product of two residuals constructed
# by using two uncorrelated random draws $\epsilon' _{1}$ and $\epsilon' _{2}$,
# and therefore, we can pull the expectation out of squares 
# 
# 
# $$
# \begin{equation*}
# E_{\epsilon _{1}}\left[ R_1(s,\epsilon'_1)\right]
# E_{\epsilon _{2}}\left[  R_1(s,\epsilon'_2)\right]
# =E_{\epsilon _{1},\epsilon _{2}}\left[ R_1(s,\epsilon'_1)R_1(s,\epsilon'_2) \right]. 
# \end{equation*}
# $$
# 
# With that result, we can re-write the objective function as just one
# expectation operator: 
# 
# $$
# \begin{equation*}
# \Xi (\theta )=E_{s,\epsilon _{1},\epsilon _{2}}\left[ \underset{\xi (\omega
# ;\theta )}{\underbrace{R_{1}\left( s,\epsilon'
# _{1}\right) R_{1}\left(
# s,\epsilon' _{2}\right) +v\left(
# R_{2}\left( s\right) \right) ^{2}}}\right] \equiv E_{\omega }\left[ \xi
# (\omega ;\theta )\right],
# \end{equation*}
# $$
# 
# where $\omega =(s,\epsilon' _{1},\epsilon' _{2})$. Therefore, we wrote the
# objective function of the deep learning method as a single expectation
# operator $E_{\omega }\left[ \xi (\omega ;\theta )\right] $ of a function $%
# \xi (\omega ;\theta )$ that depends on a vector-valued random variable $%
# \omega $. We approximate $\Xi (\theta )$ by using Monte Carlo simulation:
# 
# $$
# \begin{equation*}
# \Xi (\theta )\approx \Xi ^{n}(\theta )=\frac{1}{n}\sum_{i=1}^{n}\xi (\omega
# _{i};\theta ),
# \end{equation*}
# $$
# 
# i.e., we draw $n$ random draws of $\omega =(s,\epsilon' _{1},\epsilon' _{2})$
# and compute the average of the objective function

# In[11]:


def Ξ(n): # objective function for DL training

    # randomly drawing current states
    y = tf.random.normal(shape=(n,), stddev=σ_e_y)
    k = tf.random.uniform(shape=(n,), minval=kmin, maxval=kmax)
 
    # randomly drawing 1st realization for shocks
    e1_y = tf.random.normal(shape=(n,), stddev=σ_y)

    # randomly drawing 2nd realization for shocks
    e2_y = tf.random.normal(shape=(n,), stddev=σ_y)
    
    # residuals for n random grid points under 2 realizations of shocks
    R1_e1, R2_e1 = Residuals(e1_y, y, k)
    R1_e2, R2_e2 = Residuals(e2_y, y, k)

    # construct all-in-one expectation operator
    R_squared = R1_e1*R1_e2 + R2_e1*R2_e2 
    
    # compute average across n random draws
    return tf.reduce_mean(R_squared)


# So far, we have been using TensorFlow in the eager execution mode as if it was numpy: result of each operation is computed immediately.

# In[12]:


n = 128



# # Model training
# 
# We are now ready to perform minimization of the objective $\Xi_n$, hence to solve (or to train) the model using stochastic optimization - the stochastic gradient descent method, and in particular, its version called Adam.
# 
# Some readers might have noticed that we have never defined in this notebook a vector of coefficients $\theta$ in the code and that the objective function $\Xi_n$ doesn't seem to depend on it, but on $n$ only.
# 
# This is not quite true. The coefficients $\theta$ have actually been created, and initialized with random values. They are attached to the perceptron object and stored in the computational graph created by TensorFlow. We can easily recover these variables and print these values.

# In[13]:


θ = perceptron.trainable_variables
# print( str(θ)[:1000] ) # we truncate output


# Now it is time to choose an optimizer. In TensorFlow, the optimizer object is in charge of performing the optimization steps, given the computed gradient. For the stochastic gradient descent, the updating rule would be:
# $$\theta \leftarrow \theta(1-\lambda) - \lambda\nabla_{\theta} \Xi_n(\theta)$$
# where $\lambda$ is a learning rate.
# For Adam, the learning rate evolves over time and can be specific to each coefficient.

# In[14]:


from tensorflow.keras.optimizers import Adam, SGD
variables = perceptron.trainable_variables
optimizer = Adam()
# optimizer = SGD(λ=0.1) # SGD can be used in place of Adam


# We use the GradientTape functionality from TensorFlow to compute the gradient from the objective, and supply it to the optimizer. Also, to speed up computation, we specifically instruct TensorFlow to run each training step in a "graph" mode. This is done by using a`tf.function` decorator. As a result, all the computations will be pre-optimized in the graph and will be ran efficiently in a scalable way.

# In[15]:


@tf.function
def training_step():

    with tf.GradientTape() as tape:
        xx = Ξ(n)

    grads = tape.gradient(xx, θ)
    optimizer.apply_gradients(zip(grads,θ))

    return xx


# We repeat the training_step K times with the following function (each training step is called an epoch).

# In[16]:


def train_me(K):

    vals = []
    for k in tf.range(K):
        val = training_step()
        vals.append(val.numpy())
        
    return vals


# In[17]:


# with writer.as_default():
results = train_me(50000)


# On a modern CPU, optimization should be done within a few minutes. It would be dramatically faster on hardware adapted to deep-learning. To see how the training has performed, we can plot the empirical errors against the number of epochs.

# In[18]:


# plt.plot( np.sqrt(results) )
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()


# In the training graph, one can see that errors are decreasing until they reach $2*10^{-5}$ on average (the errors are volatile both because they depend on a specific random draw and because only two such random draws are used to approximate the expectation function). The numbers in the graph represent the mean of the squared residuals. If we take the square root, we get $5*10^{-3}$ for the $L_2$ norm, wich is still quite accurate approximation for a model with 5 states variables and a kinked solution. We show the constructed decision rule below.

# In[19]:


nk = 10
y_array = [np.log(0.8), np.log(1), np.log(1.2)]
kvec = np.linspace(kmin, kmax, nk)
for i,y in enumerate(y_array):
    yvec = np.ones(nk) * y
    ζvec, hvec = dr(yvec,kvec)
#     plt.plot(kvec, ζvec*kvec)
    
# plt.title("RBC growth (consumption decision rule)")
# plt.xlabel("$k_t$")
# plt.ylabel("$c_t$")
# plt.ylim(0.30,0.65)
# plt.grid()
# plt.savefig("consumptionrule.pdf")

ζvec0, hvec0 = dr(0*kvec,kvec)
np.savetxt("AiO_cvec.csv", ζvec0*kvec, fmt='%.6f', delimiter=",")


# In[20]:


for i,y in enumerate(y_array):
    yvec = np.ones(nk) * y
    ζvec, hvec = dr(yvec,kvec)
    knext = np.exp(yvec) * A * kvec**α + (1-δ) * kvec - ζvec*kvec
#     plt.plot(kvec, knext)
    
# plt.plot(kvec,kvec,'k--')    
# plt.title("RBC growth (capital decision rule)")
# plt.xlabel("$k_t$")
# plt.ylabel("$k_{t+1}$")
# plt.ylim(0.80,2.20)
# plt.grid()
# plt.savefig("savingrule.pdf")

knext0 = np.exp(0*kvec) * A * kvec**α + (1-δ) * kvec - ζvec0*kvec
np.savetxt("AiO_kvec.csv", knext0, fmt='%.6f', delimiter=",")


# In[21]:


print(np.log(0.8), np.exp(0.1281))

