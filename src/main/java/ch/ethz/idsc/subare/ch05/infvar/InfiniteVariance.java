// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

/** Example 5.5 p.106: Infinite Variance */
public class InfiniteVariance implements StandardModel, MonteCarloInterface {
  static final Scalar BACK = RealScalar.ZERO;
  static final Scalar END = RealScalar.ONE;
  static final Scalar PROB = RealScalar.of(.1);
  private final Tensor states = Tensors.vector(0, 1).unmodifiable();
  private final Tensor actions = Tensors.of(BACK, END).unmodifiable(); // increment
  // TODO make class for coinflip
  private final Distribution distribution = BernoulliDistribution.of(PROB);

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return isTerminal(state) ? Tensors.of(END) : actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (action.equals(BACK) && isTerminal(next))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }

  @Override // from MoveInterface
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    if (action.equals(END))
      return END; // END is used as state
    if (RandomVariate.of(distribution).equals(RealScalar.ZERO)) // TODO check if this is the model
      return END; // END is used as state
    return BACK; // BACK is used as state
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.vector(0);
  }

  /**************************************************/
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(END); // END is used as state
  }

  /**************************************************/
  @Override // from TransitionInterface
  public Tensor transitions(Tensor state, Tensor action) {
    return isTerminal(state) ? Tensors.of(state) : states();
  }

  @Override // from TransitionInterface
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return KroneckerDelta.of(state, next);
    // state == 0
    if (action.equals(END))
      return KroneckerDelta.of(END, next);
    // action == BACK
    return next.equals(BACK) ? RealScalar.ONE.subtract(PROB) : PROB;
  }

  @Override // from ActionValueInterface
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (state.equals(BACK) && action.equals(BACK))
      return PROB; // 0.1 * 1
    return RealScalar.ZERO;
  }
}
