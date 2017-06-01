# ch.ethz.idsc.subare

Java 8 implementation of algorithms, examples, and exercises from the 2nd edition (2016 draft) of

[Sutton and Barto: Reinforcement Learning](http://incompleteideas.net/sutton/book/the-book-2nd.html)

Our implementation is inspired by the 
[python code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
by Shangtong Zhang.

Our implementation is different from the reference in two aspects:

* the algorithms are implemented **separate** from the problem scenarios
* the math is in **exact** precision which reproduces symmetries in the results in case the problem features symmetries

List of algorithms:

* Iterative Policy Evaluation (parallel, from p.81)
* *Value Iteration* to determine V*(s) (parallel, from p.90)
* *Action-Value Iteration* to determine Q*(s,a) (parallel)
* First Visit Policy Evaluation (from p.100)
* Monte Carlo Exploring Starts (from p.107)
* Tabular Temporal Difference (from p.128)
* *Sarsa*: An on-policy TD control algorithm (from p.138)
* *Q-learning*: An off-policy TD control algorithm (from p.140)
* Expected Sarsa (from p.142)
* n-step Temporal Difference for estimating V(s) (from p.154)
* n-step Sarsa for estimating Q(s,a) (from p.157)
* Random-sample one-step tabular Q-planning (from p.169)


## Examples

### Repeated Prisoner's dilemma

Exact expected reward of two adversarial optimistic agents depending on their initial configuration:

![opts](https://cloud.githubusercontent.com/assets/4012178/26301502/b8663886-3ee1-11e7-8b27-41e0c5a65b79.png)

Exact expected reward of two adversarial Upper-Confidence-Bound agents depending on their initial configuration:

![ucbs](https://cloud.githubusercontent.com/assets/4012178/26301526/c738ad1c-3ee1-11e7-9438-e928fc349868.png)


### Gambler's problem

Value Iteration v(s)

![gambler_sv](https://cloud.githubusercontent.com/assets/4012178/25566784/05d63bf0-2de1-11e7-88e8-a2c485071c38.png)

<table><tr><td>
Action Value Iteration q(s,a)

![gambler_avi_iteration](https://cloud.githubusercontent.com/assets/4012178/26570813/bd1e3586-4512-11e7-90ef-070043f3968b.gif)

<td>
Optimal policy

![gambler_avi_policy](https://cloud.githubusercontent.com/assets/4012178/26570788/9be7a47e-4512-11e7-87e4-00eb7f438e12.gif)

</tr></table>

<table><tr><td>

Monte Carlo q(s,a)

![gambler_qsa_mces](https://cloud.githubusercontent.com/assets/4012178/26284839/a05e8808-3e44-11e7-80a8-3fe1f9d38246.gif)

<td>

ESarsa q(s,a)

![gambler_qsa_esarsa](https://cloud.githubusercontent.com/assets/4012178/26284843/aa6db530-3e44-11e7-8907-a856c22df3b8.gif)

<td>

QLearning q(s,a)

![gambler_qsa_qlearn](https://cloud.githubusercontent.com/assets/4012178/26284846/b4ebbdea-3e44-11e7-8ae6-7768ff96dd22.gif)

</tr></table>


### Blackjack

Monte Carlo Exploring Starts

![blackjack_mces](https://cloud.githubusercontent.com/assets/4012178/26628094/fef76442-45fc-11e7-84fb-1d2f5e9cb695.gif)

### Racetrack

![track2](https://cloud.githubusercontent.com/assets/4012178/25793771/55d5754c-33ce-11e7-8079-48e47c1f2a6d.gif)
