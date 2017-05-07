// code by jph
package ch.ethz.idsc.subare.core.mc;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepAdapter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** useful to generate a trajectory/episode that is result of taking
 * actions according to a given policy from a given start state */
public final class MonteCarloEpisode implements EpisodeInterface {
  private final MonteCarloInterface monteCarloInterface;
  private final PolicyWrap policyWrap;
  private Tensor state;

  /** @param monteCarloInterface
   * @param policyInterface
   * @param state start of episode */
  public MonteCarloEpisode(MonteCarloInterface monteCarloInterface, PolicyInterface policyInterface, Tensor state) {
    this.monteCarloInterface = monteCarloInterface;
    policyWrap = new PolicyWrap(policyInterface);
    this.state = state;
  }

  @Override
  public final StepInterface step() {
    final Tensor prev = state;
    Tensor actions = monteCarloInterface.actions(state);
    final Tensor action = policyWrap.next(prev, actions);
    final Tensor stateS = monteCarloInterface.move(state, action);
    final Scalar reward = monteCarloInterface.reward(state, action, stateS);
    state = stateS;
    return new StepAdapter(prev, action, reward, stateS);
  }

  @Override
  public final boolean hasNext() {
    return !monteCarloInterface.isTerminal(state);
  }
}
