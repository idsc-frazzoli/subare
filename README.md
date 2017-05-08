# subare

Java 8 implementation of algorithms, examples, and exercises from the 2nd edition (2016 draft) of

[Sutton and Barto: Reinforcement Learning](http://incompleteideas.net/sutton/book/the-book-2nd.html)

Our implementation is inspired by the 
[python code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
by Shangtong Zhang

Our implementation is different from the reference in two aspects

* the algorithms are implemented separate from the problem scenarios
* the math is in exact precision which reproduces symmetries in the results in case the problem features symmetries

List of algorithms:

* Iterative Policy Evaluation (parallel, from p.81)
* Value Iteration (parallel, from p.90)
* First Visit Policy Evaluation (from p.100)
* Monte Carlo Exploring Starts (from p.107)
* Tabular Temporal Difference (from p.128)
* Sarsa: An on-policy TD control algorithm (from p.138)
* Q-learning: An off-policy TD control algorithm (from p.140)
* Expected Sarsa (from p.142)


## Examples

### Repeated Prisoner's dilemma

Expected reward of two adversarial optimistic agents depending on their initial configuration:

![optimist](https://cloud.githubusercontent.com/assets/4012178/25737770/d2df9dee-3179-11e7-8fb4-0faf438cab33.png)

Expected reward of two adversarial Upper-Confidence-Bound agents depending on their initial configuration:

![ucb](https://cloud.githubusercontent.com/assets/4012178/25737893/69aabeb6-317a-11e7-9b21-73f1298cdc3d.png)


### Gambler's problem

![gambler_sv](https://cloud.githubusercontent.com/assets/4012178/25566784/05d63bf0-2de1-11e7-88e8-a2c485071c38.png)

![gambler_act](https://cloud.githubusercontent.com/assets/4012178/25566785/092e2a2e-2de1-11e7-85d8-89782c9357ab.png)

### Racetrack

![track2](https://cloud.githubusercontent.com/assets/4012178/25793771/55d5754c-33ce-11e7-8079-48e47c1f2a6d.gif)
