// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.util.Coinflip;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.num.Boole;

/** Example 6.2: Random Walk, p.133 */
class Randomwalk implements MonteCarloInterface {
  private static final Coinflip COINFLIP = Coinflip.of(RationalScalar.HALF);
  // ---
  private static final Tensor TERMINATE1 = RealScalar.ZERO; // A
  // ---
  private final Tensor states;
  private final Tensor terminate2; // A'

  /** Context:
   * Example 6.2 uses 5 non-terminating states
   * Example 7.1 uses 19 non-terminating states
   * 
   * @param numel number of non-terminating states */
  public Randomwalk(int numel) {
    states = Range.of(0, numel + 2).unmodifiable();
    terminate2 = Last.of(states);
  }

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
    return Boole.of(!isTerminal(state) && next.equals(terminate2));
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    return COINFLIP.tossHead() //
        ? state.add(RealScalar.ONE)
        : state.subtract(RealScalar.ONE);
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.vector(states().length() / 2);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) //
        || state.equals(terminate2);
  }
}
