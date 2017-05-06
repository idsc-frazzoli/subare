# subare

Java 8 implementation of algorithms, examples, and exercises from by the book

[Sutton and Barto: Reinforcement Learning](http://incompleteideas.net/sutton/book/the-book-2nd.html)

Our implementation is inspired by the 
[python code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
by Shangtong Zhang

Our implementation is different in two aspects

* the algorithms are implemented separate from the problem scenarios
* the math is in exact precision which reproduces symmetries in the results in case the problem features symmetries

Our implementation covers the algorithms

* Iterative Policy Evaluation (from p.81)
* Value iteration (from p.90)
* First visit Monte Carlo (from p.100)


## Examples

### Repeated Prisoner's dilemma

Expected average reward of two adversarial optimistic agents:

![optimist](https://cloud.githubusercontent.com/assets/4012178/25737770/d2df9dee-3179-11e7-8fb4-0faf438cab33.png)

Expected average reward of two adversarial Upper-Confidence-Bound agents:

![ucb](https://cloud.githubusercontent.com/assets/4012178/25737893/69aabeb6-317a-11e7-9b21-73f1298cdc3d.png)


### Gambler's problem

![gambler_sv](https://cloud.githubusercontent.com/assets/4012178/25566784/05d63bf0-2de1-11e7-88e8-a2c485071c38.png)

![gambler_act](https://cloud.githubusercontent.com/assets/4012178/25566785/092e2a2e-2de1-11e7-85d8-89782c9357ab.png)

