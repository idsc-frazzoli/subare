// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import java.util.Random;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

/** Example 6.7 p.143: Maximization bias
 * 
 * Credit: Hado van Hasselt (2010, 2011) */
class Maxbias implements MonteCarloInterface, StandardModel {
  static final double MEAN = -0.1;
  static final Scalar STATE_A = RealScalar.of(2);
  static final Scalar STATE_B = RealScalar.of(1);
  static final Scalar STATE_L = RealScalar.of(0);
  final Tensor states = Tensors.vector(0, 1, 2, 3).unmodifiable();
  final Tensor actionsA = Tensors.vector(-1, 1); // left, or right
  final Tensor actionsB;
  final Random random = new Random();

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

  // @Override
  // @Deprecated
  // public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
  // Tensor next = move(state, action); // deterministic
  // if (next.equals(STATE_B))
  // return gvalues.value(STATE_B); // TODO check
  // if (state.equals(STATE_B))
  // return DoubleScalar.of(MEAN);
  // // TODO <- zero?
  // return gvalues.value(state);
  // }
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
      return RealScalar.of(MEAN + random.nextGaussian());
    return RealScalar.ZERO;
  }

  /**************************************************/
  @Override
  public boolean isTerminal(Tensor state) {
    return !state.equals(STATE_A) && !state.equals(STATE_B);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return Tensors.of(STATE_A);
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (state.equals(STATE_B))
      return RealScalar.of(MEAN);
    return RealScalar.ZERO;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(move(state, action), next);
  }
}
