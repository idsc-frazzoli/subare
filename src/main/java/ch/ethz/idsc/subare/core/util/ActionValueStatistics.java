// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** class digests (s,a,r,s') and maintains a statistic to estimate
 * 
 * 1) (s,a) -> E[r]
 * 2) (s,a) -> union of move(s,a) == all possible states that can follow (s,a)
 * 3) (s,a) -> p(s'|s,a)
 * 
 * the three (estimated) functions constitute {@link ActionValueInterface}
 * 
 * (s,a,r,s') originate from episodes, or single step trials */
// TODO name of class ?
public class ActionValueStatistics implements StepDigest, ActionValueInterface {
  private final Map<Tensor, TransitionTracker> transitionTrackers = new HashMap<>();

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Tensor key = DiscreteQsa.createKey(state0, action);
    if (!transitionTrackers.containsKey(key))
      transitionTrackers.put(key, new TransitionTracker());
    transitionTrackers.get(key).digest(stepInterface);
  }

  /** @return true, if all states from model have been digested at least once
   * otherwise false */
  public boolean isComplete(DiscreteModel discreteModel) {
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Tensor key = DiscreteQsa.createKey(state, action);
        if (!transitionTrackers.containsKey(key))
          return false;
      }
    return true;
  }

  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    return transitionTrackers.get(key).expectedReward();
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    return transitionTrackers.get(key).transitions();
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    Tensor key = DiscreteQsa.createKey(state, action);
    return transitionTrackers.get(key).transitionProbability(next);
  }
}
