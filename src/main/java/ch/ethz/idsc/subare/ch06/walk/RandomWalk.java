// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Range;

/** Example 6.2: Random Walk, p.133 */
class RandomWalk implements StandardModel, MonteCarloInterface, EpisodeSupplier {
  private static final Tensor TERMINATE1 = ZeroScalar.get(); // A
  private static final Tensor TERMINATE2 = RealScalar.of(6); // A'
  // ---
  final Tensor states = Range.of(0, 7).unmodifiable();
  final Index statesIndex;

  public RandomWalk() {
    statesIndex = Index.build(states);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.vector(0);
    return Tensors.vector(-1, +1);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    Tensor next = move(state, action);
    int nextI = statesIndex.of(next);
    return reward(state, action, next).add(gvalues.get(nextI));
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (!isTerminal(state) && stateS.equals(TERMINATE2))
      return RealScalar.ONE;
    return ZeroScalar.get();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  /**************************************************/
  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    Tensor start = RealScalar.of(3);
    return new MonteCarloEpisode(this, policyInterface, start);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) || state.equals(TERMINATE2);
  }
}
