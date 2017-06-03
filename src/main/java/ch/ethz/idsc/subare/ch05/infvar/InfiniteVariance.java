// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

class InfiniteVariance extends DeterministicStandardModel implements MonteCarloInterface {
  private final Tensor states = Tensors.vector(0, 1).unmodifiable();
  final Tensor actions = Tensors.vector(0, 1).unmodifiable(); // increment
  final Index statesIndex;

  public InfiniteVariance() {
    statesIndex = Index.build(states);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return isTerminal(state) ? Tensors.of(RealScalar.ZERO) : actions;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return state.equals(RealScalar.ZERO) && action.equals(RealScalar.ONE) ? //
        RealScalar.ONE : RealScalar.ZERO;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return Tensors.vector(0);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(RealScalar.ONE);
  }

  /**************************************************/
  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }
}
