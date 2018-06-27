package ch.ethz.idsc.subare.core.td;

import java.util.Random;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Join;

public class TrueOnlineSarsa {
  private final Scalar lambda;
  private final Scalar alpha;
  private Scalar qOld = RealScalar.ZERO;
  private Scalar q;
  private Scalar q_prim;
  private Tensor x;
  private Tensor x_prim;
  private Tensor w;
  private Tensor z;
  private Scalar delta;
  private final Scalar gamma;
  private final int featureSize;
  private final int dimState;
  private final int dimAction;
  private final MonteCarloInterface mcInterface;
  private final Random rand = new Random();
  private final FeatureMapper mapper;

  public TrueOnlineSarsa(MonteCarloInterface mcInterface, Scalar lambda, Scalar alpha, Scalar gamma, FeatureMapper mapper) {
    this.mcInterface = mcInterface;
    this.lambda = lambda;
    this.alpha = alpha;
    this.gamma = gamma;
    this.mapper = mapper;
    dimState = mcInterface.states().get(0).length();
    dimAction = mcInterface.actions(mcInterface.states().get(0)).get(0).length();
    featureSize = mapper.getFeatureSize();
    z = Array.zeros(featureSize);
    w = Array.zeros(featureSize);
    GlobalAssert.that(lambda.Get().number().doubleValue() >= 0.0);
    GlobalAssert.that(lambda.Get().number().doubleValue() <= 1.0);
    GlobalAssert.that(alpha.Get().number().doubleValue() >= 0.0);
  }

  public void update(Scalar r, Tensor s_prim, Tensor a_prim) {
    x_prim = mapper.getFeature(Join.of(s_prim, a_prim));
    q = w.dot(x).Get();
    q_prim = w.dot(x_prim).Get();
    delta = r.add(gamma.multiply(q_prim)).subtract(q);
    z = z.multiply(lambda).multiply(gamma).add(x.multiply(RealScalar.ONE.subtract(alpha.multiply(gamma).multiply(lambda).multiply(z.dot(x).Get()))));
    w = w.add(z.multiply(alpha).multiply(delta.add(q.subtract(qOld)))).subtract(x.multiply(alpha).multiply(q.subtract(qOld)));
    qOld = q_prim;
    x = x_prim;
  }

  public Tensor getEGreedyAction(Tensor state, Scalar epsilon) {
    if (rand.nextFloat() > epsilon.number().doubleValue()) {
      return getGreedyAction(state);
    }
    int index = rand.nextInt(mcInterface.actions(state).length());
    return mcInterface.actions(state).get(index);
  }

  public Tensor getGreedyAction(Tensor state) {
    double max = Double.NEGATIVE_INFINITY;
    Tensor bestAction = Tensors.empty();
    for (Tensor action : mcInterface.actions(state)) {
      double current = mapper.getFeature(Join.of(state, (Tensor) action)).dot(w).Get().number().doubleValue();
      if (current > max) {
        bestAction = action;
        max = current;
      }
    }
    return bestAction;
  }

  public void executeEpisode(Scalar epsilon) {
    // getting random index for startState
    int index = rand.nextInt(mcInterface.startStates().length());
    Tensor state = mcInterface.startStates().get(index);
    Tensor stateOld;
    Tensor action = getEGreedyAction(state, epsilon);
    Tensor actionOld;
    Scalar reward;
    // init every episode again
    x = mapper.getFeature(Join.of(state, action));
    qOld = RealScalar.ZERO;
    z = Array.zeros(featureSize);
    // run through episode
    while (!mcInterface.isTerminal(state)) {
      stateOld = state;
      actionOld = action;
      state = mcInterface.move(stateOld, actionOld);
      reward = mcInterface.reward(stateOld, actionOld, state);
      // System.out.println("from state " + stateOld + " to " + state + " with action " + actionOld + " reward: " + reward);
      action = getEGreedyAction(state, epsilon);
      update(reward, state, action);
    }
  }

  public void printValues() {
    System.out.println("Values for all state-action pairs:");
    for (Tensor state : mcInterface.states()) {
      for (Tensor action : mcInterface.actions(state)) {
        System.out.println(state + " -> " + action + " " + mapper.getFeature(Join.of(state, action)).dot(w));
      }
    }
  }

  public void printPolicy() {
    System.out.println("Greedy action to each state");
    for (Tensor state : mcInterface.states()) {
      System.out.println(state + " -> " + getGreedyAction(state));
    }
  }

  public Tensor getW() {
    return w;
  }
}
