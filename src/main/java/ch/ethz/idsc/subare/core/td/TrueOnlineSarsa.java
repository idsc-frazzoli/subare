// code by fluric
package ch.ethz.idsc.subare.core.td;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clip;

/** implementation of box "True Online Sarsa(lambda) for estimating w'x approx. q_pi or q_*
 * 
 * in Section 12.8, p.309 */
public class TrueOnlineSarsa implements DiscreteQsaSupplier {
  private final Random random = new Random();
  private final MonteCarloInterface monteCarloInterface;
  private final Scalar gamma;
  private final FeatureMapper featureMapper;
  private final LearningRate learningRate;
  // ---
  private final Scalar gamma_lambda;
  private Scalar alpha_gamma_lambda;
  private final int featureSize;
  // private final int dimState;
  // private final int dimAction;
  // ---
  private Scalar qOld = RealScalar.ZERO;
  private Scalar q;
  private Scalar q_prime;
  private Tensor x;
  private Tensor x_prime;
  private Tensor w;
  private Tensor z;
  private Scalar delta;

  /** @param monteCarloInterface
   * @param lambda in [0, 1]
   * @param learningRate
   * @param featureMapper
   * @param init */
  public TrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Scalar init) {
    this.monteCarloInterface = monteCarloInterface;
    this.learningRate = learningRate;
    Clip.unit().requireInside(lambda);
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, lambda);
    // dimState = monteCarloInterface.states().get(0).length();
    // dimAction = monteCarloInterface.actions(monteCarloInterface.states().get(0)).get(0).length();
    featureSize = featureMapper.getFeatureSize();
    z = Array.zeros(featureSize);
    w = Tensors.vector(v -> init, featureSize);
  }

  public TrueOnlineSarsa(MonteCarloInterface mcInterface, Scalar lambda, LearningRate learningRate, FeatureMapper mapper) {
    this(mcInterface, lambda, learningRate, mapper, RealScalar.ZERO); // exact precision intentional
  }

  private void update(Scalar reward, Tensor s, Tensor s_prime, Tensor a_prime) {
    Tensor stateActionPair = StateAction.key(s_prime, a_prime);
    StepInterface stepInterface = new StepAdapter(s, a_prime, reward, s_prime);
    Scalar alpha = learningRate.alpha(stepInterface);
    learningRate.digest(stepInterface);
    alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    x_prime = featureMapper.getFeature(stateActionPair);
    q = w.dot(x).Get();
    q_prime = w.dot(x_prime).Get();
    delta = reward.add(gamma.multiply(q_prime)).subtract(q);
    z = z.multiply(gamma_lambda).add(x.multiply(RealScalar.ONE.subtract(alpha_gamma_lambda.multiply(z.dot(x).Get()))));
    // ---
    Scalar q_qOld = q.subtract(qOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(q_qOld)));
    Tensor scalex = x.multiply(alpha.multiply(q_qOld));
    w = w.add(scalez).subtract(scalex);
    qOld = q_prime;
    x = x_prime;
  }

  /** Returns the epsilon greedy action.
   * With probability epsilon a random action is chosen. In the other case the best
   * (greedy) action is taken with equal probability when several best actions.
   * @param state
   * @param epsilon
   * @return */
  private Tensor getEGreedyAction(Tensor state, Scalar epsilon) {
    Tensor actions = monteCarloInterface.actions(state);
    if (random.nextFloat() > epsilon.number().doubleValue()) {
      actions = getGreedyAction(state);
    }
    int index = random.nextInt(actions.length());
    return actions.get(index);
  }

  /** Returns the best action according to the current state-action values. In case
   * of several best actions within a tolerance, all the best actions are returned.
   * 
   * @param state
   * @return */
  private Tensor getGreedyAction(Tensor state) {
    double max = Double.NEGATIVE_INFINITY;
    Tensor bestActions = Tensors.empty();
    for (Tensor action : monteCarloInterface.actions(state)) {
      Tensor stateActionPair = StateAction.key(state, action);
      double current = featureMapper.getFeature(stateActionPair).dot(w).Get().number().doubleValue();
      if (Math.abs(current - max) < 1e-8) {
        bestActions.append(action);
      } else if (current > max) {
        max = current;
        bestActions = Tensors.of(action);
      }
    }
    return bestActions;
  }

  public void executeEpisode(Scalar epsilon) {
    // getting random index for startState
    Tensor states = monteCarloInterface.startStates();
    Tensor state = states.get(random.nextInt(states.length()));
    Tensor action = getEGreedyAction(state, epsilon);
    executeEpisode(epsilon, state, action);
  }

  public void executeEpisode(Scalar epsilon, Tensor state, Tensor action) {
    // init every episode again
    Tensor stateActionPair = StateAction.key(state, action);
    x = featureMapper.getFeature(stateActionPair);
    qOld = RealScalar.ZERO;
    z = Array.zeros(featureSize);
    // run through episode
    while (!monteCarloInterface.isTerminal(state)) {
      Tensor stateOld = state;
      Tensor actionOld = action;
      state = monteCarloInterface.move(stateOld, actionOld);
      Scalar reward = monteCarloInterface.reward(stateOld, actionOld, state);
      // System.out.println("from state " + stateOld + " to " + state + " with action " + actionOld + " reward: " + reward);
      action = getEGreedyAction(state, epsilon);
      // System.out.println(action);
      update(reward, stateOld, state, action);
    }
  }

  public void executeBatch(Scalar epsilon) {
    List<Tensor> list = new ArrayList<>();
    for (Tensor state : monteCarloInterface.startStates())
      for (Tensor action : monteCarloInterface.actions(state))
        list.add(Tensors.of(state, action));
    Collections.shuffle(list);
    for (Tensor stateActionPair : list)
      executeEpisode(epsilon, stateActionPair.get(0), stateActionPair.get(1));
  }

  public void printValues() {
    System.out.println("Values for all state-action pairs:");
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state))
        System.out.println(state + " -> " + action + " " + featureMapper.getFeature(Join.of(state, action)).dot(w));
  }

  /** Returns the Qsa according to the current feature weights.
   * Only use this function, when the state-action space is small enough. */
  public DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
    for (Tensor state : monteCarloInterface.states()) {
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        qsa.assign(state, action, featureMapper.getFeature(stateActionPair).dot(w).Get());
      }
    }
    return qsa;
  }

  public void printPolicy() {
    System.out.println("Greedy action to each state");
    for (Tensor state : monteCarloInterface.states())
      System.out.println(state + " -> " + getGreedyAction(state));
  }

  public Tensor getW() {
    return w.unmodifiable();
  }
}
