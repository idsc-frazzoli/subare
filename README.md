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
* Double Q-Learning (from p.145)
* n-step Temporal Difference for estimating V(s) (from p.154)
* n-step Sarsa for estimating Q(s,a) (from p.157)
* Random-sample one-step tabular Q-planning (from p.169)

## Examples from the book

### 4.1 Gridworld

Action Value Iteration q(s,a)

### 4.2: Jack's car rental

Value Iteration v(s)

![gambler_avi](https://cloud.githubusercontent.com/assets/4012178/26673482/5a11e616-46bd-11e7-8c52-376acac21fa8.gif)

### 4.4: Gambler's problem

Value Iteration v(s)

![gambler_sv](https://cloud.githubusercontent.com/assets/4012178/25566784/05d63bf0-2de1-11e7-88e8-a2c485071c38.png)

Action Value Iteration and optimal policy

![gambler_avi](https://cloud.githubusercontent.com/assets/4012178/26673482/5a11e616-46bd-11e7-8c52-376acac21fa8.gif)

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


### 5.1 Blackjack

Monte Carlo Exploring Starts

![blackjack_mces](https://cloud.githubusercontent.com/assets/4012178/26628094/fef76442-45fc-11e7-84fb-1d2f5e9cb695.gif)

### 5.2 Wireloop

<table><tr><td>

AV-Iteration

![wire4_avi_iteration](https://cloud.githubusercontent.com/assets/4012178/26669492/63ce19ea-46ae-11e7-806c-849eba3da453.gif)

<td>

Q-Learning

![wire4_ql](https://cloud.githubusercontent.com/assets/4012178/26669495/661de338-46ae-11e7-99b3-cc69d7d853a7.gif)

<td>

Monte Carlo

![wire4_qsa_mces](https://cloud.githubusercontent.com/assets/4012178/26669499/698b4da8-46ae-11e7-9a8e-5ec837a2304a.gif)

<td>

E-Sarsa

![wire4_qsa_esarsa](https://cloud.githubusercontent.com/assets/4012178/26669504/6bf3bb20-46ae-11e7-9464-6dfb029a82db.gif)

<td>

Sarsa

![wire4_qsa_sarsa](https://cloud.githubusercontent.com/assets/4012178/26669505/6c18cdca-46ae-11e7-9b39-4086e6e1b25a.gif)

</tr></table>

### 5.8 Racetrack

paths obtained using value iteration

<table><tr><td valign="top">

track 1

![track1](https://cloud.githubusercontent.com/assets/4012178/26668651/01d5ff76-46ab-11e7-9332-7aadecd5923e.gif)

<td><td><td valign="top">

track 2

![track2](https://cloud.githubusercontent.com/assets/4012178/26668652/0417e402-46ab-11e7-884f-c95471775c9b.gif)

</tr></table>

---

## Additional Examples

### Repeated Prisoner's dilemma

Exact expected reward of two adversarial optimistic agents depending on their initial configuration:

![opts](https://cloud.githubusercontent.com/assets/4012178/26301502/b8663886-3ee1-11e7-8b27-41e0c5a65b79.png)

Exact expected reward of two adversarial Upper-Confidence-Bound agents depending on their initial configuration:

![ucbs](https://cloud.githubusercontent.com/assets/4012178/26301526/c738ad1c-3ee1-11e7-9438-e928fc349868.png)

## Dependencies

`subare` requires the library `ch.ethz.idsc.tensor`
