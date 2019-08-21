// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.NormalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

/** Example 6.7 p.134: Maximization bias
 * 
 * Credit: Hado van Hasselt (2010, 2011) */
public class Maxbias implements StandardModel, MonteCarloInterface {
  static final Scalar MEAN = RealScalar.of(-0.1);
  static final Scalar STATE_A = RealScalar.of(2);
  static final Scalar STATE_B = RealScalar.of(1);
  static final Scalar STATE_L = RealScalar.of(0);
  final Tensor states = Tensors.vector(0, 1, 2, 3).unmodifiable();
  final Tensor actionsA = Tensors.vector(-1, 1); // left, or right
  final Tensor actionsB;
  final Distribution distribution = NormalDistribution.of(MEAN, RealScalar.ONE);

  public Maxbias(int choices) {
    actionsB = Range.of(0, choices).unmodifiable();
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (state.equals(STATE_A))
      return actionsA;
    if (state.equals(STATE_B))
      return actionsB;
    return Tensors.vector(0);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(STATE_A))
      return state.add(action);
    if (state.equals(STATE_B))
      return STATE_L;
    return state;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (state.equals(STATE_B))
      return RandomVariate.of(distribution);
    return RealScalar.ZERO;
  }

  /**************************************************/
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return !state.equals(STATE_A) && !state.equals(STATE_B);
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(STATE_A);
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (state.equals(STATE_B))
      return MEAN;
    return RealScalar.ZERO;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    // TODO this implementation does not make sense
    if (move(state, action).equals(next))
      return KroneckerDelta.of(move(state, action), next);
    throw TensorRuntimeException.of(state, action, next);
  }
}
