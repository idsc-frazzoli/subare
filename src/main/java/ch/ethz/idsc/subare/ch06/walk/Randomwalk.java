// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;

/** Example 6.2: Random Walk, p.133 */
class Randomwalk implements MonteCarloInterface {
  private static final Distribution COINFLIPPING = BernoulliDistribution.of(RationalScalar.of(1, 2));
  // ---
  private static final Tensor TERMINATE1 = RealScalar.ZERO; // A
  private static final Tensor TERMINATE2 = RealScalar.of(6); // A'
  // ---
  private final Tensor states = Range.of(0, 7).unmodifiable();

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return Tensors.vector(0);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (!isTerminal(state) && next.equals(TERMINATE2))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    return RandomVariate.of(COINFLIPPING).equals(RealScalar.ZERO) ? //
        state.add(RealScalar.ONE) : state.subtract(RealScalar.ONE);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return Tensors.vector(3);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) || state.equals(TERMINATE2);
  }
}
