# subare

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
* Random-sample one-step tabular Q-planning (from p.169)


## Examples

### Repeated Prisoner's dilemma

Expected reward of two adversarial optimistic agents depending on their initial configuration:

![optimist](https://cloud.githubusercontent.com/assets/4012178/25737770/d2df9dee-3179-11e7-8fb4-0faf438cab33.png)

Expected reward of two adversarial Upper-Confidence-Bound agents depending on their initial configuration:

![ucb](https://cloud.githubusercontent.com/assets/4012178/25737893/69aabeb6-317a-11e7-9b21-73f1298cdc3d.png)


### Gambler's problem

![gambler_sv](https://cloud.githubusercontent.com/assets/4012178/25566784/05d63bf0-2de1-11e7-88e8-a2c485071c38.png)

![gambler_act](https://cloud.githubusercontent.com/assets/4012178/25566785/092e2a2e-2de1-11e7-85d8-89782c9357ab.png)

Exact function q(s,a)

![gambler_qsa_avi](https://cloud.githubusercontent.com/assets/4012178/26282929/2edff964-3e1c-11e7-996a-4450070df589.png)

Monte Carlo q(s,a)

![gambler_qsa_mces](https://cloud.githubusercontent.com/assets/4012178/26282931/40fd07cc-3e1c-11e7-9831-ccb707a44a96.gif)

ESarsa q(s,a)

![gambler_qsa_esarsa](https://cloud.githubusercontent.com/assets/4012178/26282932/4c85ec12-3e1c-11e7-8bd5-1fe4f2ea2e7e.gif)

QLearning q(s,a)

![gambler_qsa_qlearn](https://cloud.githubusercontent.com/assets/4012178/26282933/5721d5d2-3e1c-11e7-9a59-183f0ad350e7.gif)


### Racetrack

![track2](https://cloud.githubusercontent.com/assets/4012178/25793771/55d5754c-33ce-11e7-8079-48e47c1f2a6d.gif)
