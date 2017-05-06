// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepAdapter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Array;

public final class MonteCarloEpisode implements EpisodeInterface {
  private final MonteCarloInterface monteCarloInterface;
  private final PolicyInterface policyInterface;
  private final Random random = new Random();
  private Tensor state;

  /** @param monteCarloInterface
   * @param policyInterface
   * @param state start of episode */
  public MonteCarloEpisode(MonteCarloInterface monteCarloInterface, PolicyInterface policyInterface, Tensor state) {
    this.monteCarloInterface = monteCarloInterface;
    this.policyInterface = policyInterface;
    this.state = state;
  }

  @Override
  public final StepInterface step() {
    final Tensor prev = state;
    Tensor actions = monteCarloInterface.actions(state);
    Index actionIndex = Index.build(actions);
    Tensor prob = Array.zeros(actions.length());
    for (Tensor action : actions)
      prob.set(policyInterface.policy(state, action), actionIndex.of(action));
    prob = Accumulate.of(prob);
    double threshold = random.nextDouble();
    int index = 0;
    for (; index < prob.length(); ++index)
      if (Scalars.lessThan(DoubleScalar.of(threshold), prob.Get(index)))
        break;
    final Tensor action = actions.get(index);
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
