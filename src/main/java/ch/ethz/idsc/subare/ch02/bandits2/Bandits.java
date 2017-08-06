// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.Expectation;
import ch.ethz.idsc.tensor.pdf.NormalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Variance;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** "A k-armed Bandit Problem"
 * Section 2.1 p.28 */
class Bandits implements StandardModel, MonteCarloInterface {
  static final Tensor START = RealScalar.ZERO;
  static final Tensor END = RealScalar.ONE;
  // ---
  private final List<Distribution> distributions = new ArrayList<>();

  /** @param k number of arms of bandit */
  public Bandits(int k) {
    Distribution STANDARD = NormalDistribution.standard();
    Tensor data = RandomVariate.of(STANDARD, k);
    Scalar mean = (Scalar) Mean.of(data);
    Tensor temp = data.map(x -> x.subtract(mean)).unmodifiable();
    Tensor prep = temp.divide(Sqrt.of((Scalar) Variance.ofVector(temp)));
    for (int index = 0; index < k; ++index)
      distributions.add(NormalDistribution.of(prep.Get(index), RealScalar.ONE));
  }

  @Override
  public Tensor states() {
    return Tensors.of(START, END);
  }

  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.vector(0);
    return Range.of(0, distributions.size());
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    return END;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    int index = action.Get().number().intValue();
    return RandomVariate.of(distributions.get(index));
  }

  /**************************************************/
  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(END);
  }

  @Override
  public Tensor startStates() {
    return Tensors.of(START);
  }

  /**************************************************/
  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(END);
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(next, END);
  }

  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    int index = action.Get().number().intValue();
    return Expectation.mean(distributions.get(index));
  }
}
