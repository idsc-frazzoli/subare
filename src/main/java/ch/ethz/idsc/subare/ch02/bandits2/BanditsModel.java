// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import java.util.List;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.api.TensorUnaryOperator;
import ch.ethz.idsc.tensor.nrm.Normalize;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.NormalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.StandardDeviation;

/** "A k-armed Bandit Problem"
 * Section 2.1 p.28 */
/* package */ class BanditsModel implements StandardModel, MonteCarloInterface {
  private static final TensorUnaryOperator NORMALIZE = Normalize.with(StandardDeviation::ofVector);
  /** state before choosing bandit */
  static final Tensor START = RealScalar.ZERO;
  /** terminal state after choosing bandit */
  static final Tensor END = RealScalar.ONE;
  // ---
  private final List<Distribution> distributions;
  private final Tensor actions;

  /** @param k number of arms of bandit */
  public BanditsModel(int k) {
    Tensor data = RandomVariate.of(NormalDistribution.standard(), k);
    Scalar mean = (Scalar) Mean.of(data);
    Tensor prep = NORMALIZE.apply(data.map(x -> x.subtract(mean)));
    distributions = prep.stream() //
        .map(Scalar.class::cast) //
        .map(scalar -> NormalDistribution.of(scalar, RealScalar.ONE)) //
        .collect(Collectors.toList());
    actions = Range.of(0, k).unmodifiable();
  }

  @Override // from StateActionModel
  public Tensor states() {
    return Tensors.of(START, END);
  }

  @Override // from StateActionModel
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.vector(0);
    return actions;
  }

  @Override // from DiscreteModel
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override // from MoveInterface
  public Tensor move(Tensor state, Tensor action) {
    return END;
  }

  @Override // from RewardInterface
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    int index = Scalars.intValueExact((Scalar) action);
    return RandomVariate.of(distributions.get(index));
  }

  /**************************************************/
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(END);
  }

  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(START);
  }

  /**************************************************/
  @Override // from TransitionInterface
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(END);
  }

  @Override // from TransitionInterface
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(next, END);
  }

  @Override // from ActionValueInterface
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    int index = Scalars.intValueExact((Scalar) action);
    return Mean.of(distributions.get(index));
  }
}
