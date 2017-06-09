// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.TerminalInterface;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;

/** class digests (s,a,r,s') and maintains a statistic to estimate
 * 
 * 1) (s,a) -> E[r]
 * 2) (s,a) -> union of move(s,a) == all possible states that can follow (s,a)
 * 3) (s,a) -> p(s'|s,a)
 * 
 * the three (estimated) functions constitute {@link ActionValueInterface}
 * 
 * (s,a,r,s') originate from episodes, or single step trials */
public class ActionValueStatistics implements StepDigest, EpisodeDigest, ActionValueInterface {
  private final Map<Tensor, TransitionTracker> transitionTrackers = new HashMap<>();
  private final DiscreteModel discreteModel;

  public ActionValueStatistics(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
    if (discreteModel instanceof TerminalInterface) {
      TerminalInterface terminalInterface = (TerminalInterface) discreteModel;
      for (Tensor state : discreteModel.states())
        if (terminalInterface.isTerminal(state))
          digestTerminal(state);
    }
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Tensor key = DiscreteQsa.createKey(state0, action);
    if (!transitionTrackers.containsKey(key))
      transitionTrackers.put(key, new TransitionTracker());
    transitionTrackers.get(key).digest(stepInterface);
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
    StepInterface stepInterface = null;
    while (episodeInterface.hasNext()) {
      stepInterface = episodeInterface.step();
      digest(stepInterface);
    }
    if (stepInterface == null)
      throw new RuntimeException(); // episode start should not be terminal
    // TODO this line is optional if the constructor does the job...
    digestTerminal(stepInterface.nextState()); // terminal state
  }

  /**************************************************/
  /** build a step interface for the transition from the terminal state into the terminal state
   * 
   * @param state */
  public void digestTerminal(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    if (actions.length() != 1)
      // terminal state should only allow 1 action
      throw TensorRuntimeException.of(state, actions);
    Tensor action = actions.get(0);
    // TODO can test via reward interface if model also returns reward == 0
    Scalar reward = RealScalar.ZERO;
    digest(new StepAdapter(state, action, reward, state));
  }

  /** @return true, if all states from model have been digested at least once
   * otherwise false */
  public boolean isComplete() {
    return coverage().equals(RealScalar.ONE);
  }

  /** @return ratio of (state, action) pairs visited vs total */
  public Scalar coverage() {
    int num = 0;
    int den = 0;
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Tensor key = DiscreteQsa.createKey(state, action);
        num += transitionTrackers.containsKey(key) ? 1 : 0;
        ++den;
      }
    return RationalScalar.of(num, den);
  }

  /**************************************************/
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
