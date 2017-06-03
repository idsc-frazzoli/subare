// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;

/** Example 6.2: Random Walk, p.133 */
class Randomwalk implements //
    MonteCarloInterface, EpisodeSupplier {
  private static final Tensor TERMINATE1 = RealScalar.ZERO; // A
  private static final Tensor TERMINATE2 = RealScalar.of(6); // A'
  // ---
  private final Tensor states = Range.of(0, 7).unmodifiable();
  Random random = new Random();

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return Tensors.vector(0);
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    if (!isTerminal(state) && stateS.equals(TERMINATE2))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    return random.nextBoolean() ? state.add(RealScalar.ONE) : state.subtract(RealScalar.ONE);
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

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }
}
