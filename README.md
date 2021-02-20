# ch.ethz.idsc.subare <a href="https://travis-ci.org/idsc-frazzoli/subare"><img src="https://travis-ci.org/idsc-frazzoli/subare.svg?branch=master" alt="Build Status"></a>

Library for reinforcement learning in Java, version `0.3.9`

Repository includes algorithms, examples, and exercises from the 2nd edition of [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton, and Andrew G. Barto.

Our implementation is inspired by the
[python code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
by Shangtong Zhang, but differs from the reference in two aspects:

* the algorithms are implemented **separate** from the problem scenarios
* the math is in **exact** precision which reproduces symmetries in the results in case the problem features symmetries

## Algorithms

* Iterative Policy Evaluation (parallel, in 4.1, p.59)
* *Value Iteration* to determine V*(s) (parallel, in 4.4, p.65)
* *Action-Value Iteration* to determine Q*(s,a) (parallel)
* First Visit Policy Evaluation (in 5.1, p.74)
* Monte Carlo Exploring Starts (in 5.3, p.79)
* Contant-alpha Monte Carlo
* Tabular Temporal Difference (in 6.1, p.96)
* *Sarsa*: An on-policy TD control algorithm (in 6.4, p.104)
* *Q-learning*: An off-policy TD control algorithm (in 6.5, p.105)
* Expected Sarsa (in 6.6, p.107)
* Double Sarsa, Double Expected Sarsa, Double Q-Learning (in 6.7, p.109)
* n-step Temporal Difference for estimating V(s) (in 7.1, p.115)
* n-step Sarsa, n-step Expected Sarsa, n-step Q-Learning (in 7.2, p.118)
* Random-sample one-step tabular Q-planning (parallel, in 8.1, p.131)
* Tabular Dyna-Q (in 8.2, p.133)
* Prioritized Sweeping (in 8.4, p.137)
* Semi-gradient Tabular Temporal Difference (in 9.3, p.164)
* True Online Sarsa (in 12.8, p.309)

## Gallery

<table>
<tr>
<td>

![prisonersdilemma](https://user-images.githubusercontent.com/4012178/49791508-ed856500-fd30-11e8-960b-5a90f7ebb638.png)

Prisoner's Dilemma

<td>

![gambler_exact](https://user-images.githubusercontent.com/4012178/50052035-b7275b80-011d-11e9-8ea2-b466b90fb349.png)

Exact Gambler

</tr>
</table>

## Examples

### 4.1 Gridworld

<table><tr>

<td valign="top">

AV-Iteration q(s,a)

![gridworld_qsa_avi](https://cloud.githubusercontent.com/assets/4012178/26762465/36ac9224-4943-11e7-8fcb-d543d1766aa9.gif)

<td>

TabularQPlan

![gridworld_qsa_rstqp](https://cloud.githubusercontent.com/assets/4012178/26762466/36ae79a4-4943-11e7-9516-cdf8ca9d9c4f.gif)

<td>

Monte Carlo

![gridworld_qsa_mces](https://cloud.githubusercontent.com/assets/4012178/26762469/36af0784-4943-11e7-91ce-89f86afff7a2.gif)

</tr><tr>

<td>

Q-Learning

![gridworld_qsa_qlearning](https://cloud.githubusercontent.com/assets/4012178/26762470/36af4302-4943-11e7-8891-6fdaf95b912b.gif)

<td>

Expected-Sarsa

![gridworld_qsa_expected](https://cloud.githubusercontent.com/assets/4012178/26762468/36aedaac-4943-11e7-998d-df150fe0eca6.gif)

<td>

Sarsa

![gridworld_qsa_original](https://cloud.githubusercontent.com/assets/4012178/26762467/36ae8656-4943-11e7-8d9e-e17819c1e54b.gif)

</tr><tr>

<td>

3-step Q-Learning

![gridworld_qsa_qlearning3](https://user-images.githubusercontent.com/4012178/26866445-6aabcb88-4b61-11e7-8b04-b21613db2f65.gif)

<td>

3-step E-Sarsa

![gridworld_qsa_expected3](https://user-images.githubusercontent.com/4012178/26866446-6ad0e1ca-4b61-11e7-897f-2831e755844b.gif)

<td>

3-step Sarsa

![gridworld_qsa_original3](https://user-images.githubusercontent.com/4012178/26866447-6ad0ecec-4b61-11e7-88d2-bf6cba11c245.gif)

</tr><tr>

<td>

OTrue Online Sarsa

![gridworld_tos_original](https://user-images.githubusercontent.com/4012178/43031808-b40012b4-8ca9-11e8-9539-9fd66f9e8ba0.gif)

<td>

ETrue Online Sarsa

![gridworld_tos_expected](https://user-images.githubusercontent.com/4012178/43031809-b41b8558-8ca9-11e8-9c2c-0d514e7b1504.gif)

<td>

QTrue Online Sarsa

![gridworld_tos_qlearning](https://user-images.githubusercontent.com/4012178/43031807-b3e5f7a8-8ca9-11e8-8650-0d45638bfe5b.gif)

</tr></table>


### 4.2: Jack's car rental

Value Iteration v(s)

![carrental_vi_true](https://cloud.githubusercontent.com/assets/4012178/26762456/0d5439fe-4943-11e7-91a2-d0663484690c.gif)

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

![wire5_avi](https://cloud.githubusercontent.com/assets/4012178/26762420/588aeef0-4942-11e7-97bc-6b25ce4a20d9.gif)

<td>

TabularQPlan

![wire5_qsa_rstqp](https://cloud.githubusercontent.com/assets/4012178/26762437/cf460cbe-4942-11e7-8d5a-74af0157935d.gif)

<td>

Q-Learning

![wire5_qsa_qlearning](https://cloud.githubusercontent.com/assets/4012178/26762426/8aad7696-4942-11e7-89a6-d8279361c3eb.gif)

<td>

E-Sarsa

![wire5_qsa_expected](https://cloud.githubusercontent.com/assets/4012178/26762428/a330a17a-4942-11e7-9b8d-4d2bd5ab957a.gif)

<td>

Sarsa

![wire5_qsa_original](https://cloud.githubusercontent.com/assets/4012178/26762745/a247351c-4947-11e7-81b4-a5e810dd8661.gif)

<td>

Monte Carlo

![wire5_mces](https://cloud.githubusercontent.com/assets/4012178/26762436/bda3717c-4942-11e7-8339-b58b480cf69f.gif)

</tr></table>

### 5.8 Racetrack

paths obtained using value iteration

<table><tr><td valign="top">

track 1

![track1](https://cloud.githubusercontent.com/assets/4012178/26668651/01d5ff76-46ab-11e7-9332-7aadecd5923e.gif)

<td valign="top">

track 2

![track2](https://cloud.githubusercontent.com/assets/4012178/26668652/0417e402-46ab-11e7-884f-c95471775c9b.gif)

</tr></table>

### 6.5 Windygrid

<table><tr><td>

Action Value Iteration

![windygrid_qsa_avi](https://cloud.githubusercontent.com/assets/4012178/26816031/ebeebff2-4a8f-11e7-8bce-2d1dfa29a5a7.gif)

<td>

TabularQPlan

![windygrid_qsa_rstqp](https://cloud.githubusercontent.com/assets/4012178/26816030/ebee6f5c-4a8f-11e7-9416-37b2d30e178f.gif)

</tr></table>


### 6.6 Cliffwalk

<table><tr><td>

Action Value Iteration

![cliffwalk_qsa_avi](https://cloud.githubusercontent.com/assets/4012178/26815999/c1c86278-4a8f-11e7-834f-89a1b7df7001.gif)

<td>

Q-Learning

![cliffwalk_qsa_qlearning](https://cloud.githubusercontent.com/assets/4012178/26815998/c1c60776-4a8f-11e7-9437-65f151f3deb0.gif)

<td>

TabularQPlan

![cliffwalk_qsa_rstqp](https://cloud.githubusercontent.com/assets/4012178/26816000/c1c96880-4a8f-11e7-9f64-95768baebc10.gif)

<td>

Expected Sarsa

![cliffwalk_qsa_expected](https://cloud.githubusercontent.com/assets/4012178/26816002/c1cd225e-4a8f-11e7-8285-3682e4ba9268.gif)

</tr></table>


### 8.1 Dynamaze

<table><tr><td>

Action Value Iteration

![maze5_qsa_avi](https://user-images.githubusercontent.com/4012178/27436123-8b2578a6-575e-11e7-8edb-5ac41405f4da.gif)

<td>

Prioritized sweeping

![maze2_ps_qlearning](https://user-images.githubusercontent.com/4012178/27436055-4cd6ec42-575e-11e7-95bc-2708a2905822.gif)

</tr></table>

---

## Additional Examples

### Repeated Prisoner's dilemma

Exact expected reward of two adversarial optimistic agents depending on their initial configuration:

![opts](https://cloud.githubusercontent.com/assets/4012178/26301502/b8663886-3ee1-11e7-8b27-41e0c5a65b79.png)

Exact expected reward of two adversarial Upper-Confidence-Bound agents depending on their initial configuration:

![ucbs](https://cloud.githubusercontent.com/assets/4012178/26301526/c738ad1c-3ee1-11e7-9438-e928fc349868.png)

## Integration

Specify `dependency` and `repository` of the tensor library in the `pom.xml` file of your maven project:

```xml
<dependencies>
  <dependency>
    <groupId>ch.ethz.idsc</groupId>
    <artifactId>subare</artifactId>
    <version>0.3.9</version>
  </dependency>
</dependencies>

<repositories>
  <repository>
    <id>subare-mvn-repo</id>
    <url>https://raw.github.com/idsc-frazzoli/subare/mvn-repo/</url>
    <snapshots>
      <enabled>true</enabled>
      <updatePolicy>always</updatePolicy>
    </snapshots>
  </repository>
</repositories>
```
The source code is attached to every release.

## Contributors

Jan Hakenberg, Christian Fluri

## Publications

* [*Learning to Operate a Fleet of Cars*](https://www.research-collection.ethz.ch/handle/20.500.11850/304517)
by Christian Fluri, Claudio Ruch, Julian Zilly, Jan Hakenberg, and Emilio Frazzoli

## References

* [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
by Richard S. Sutton, and Andrew G. Barto
