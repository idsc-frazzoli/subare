// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Min;

/** Gambler's problem:
 * an action defines the amount of coins to bet
 * the action has to be non-zero unless the capital == 0
 * or the terminal cash has been reached */
class Gambler implements StandardModel {
  // TODO report bug in STZ code
  // TODO report mistake in book
  final Tensor states;
  final Index statesIndex;
  final Scalar TERMINAL_W;
  final Scalar P_win;

  /** @param P_win probabilty of winning a coin toss */
  public Gambler(int length, Scalar P_win) {
    states = Range.of(0, length + 1).unmodifiable();
    statesIndex = Index.build(states);
    TERMINAL_W = (Scalar) Last.of(states);
    this.P_win = P_win;
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (state.equals(ZeroScalar.get()) || state.equals(TERMINAL_W))
      return Tensors.of(ZeroScalar.get());
    Scalar stateS = state.Get();
    return Range.of(1, Min.of(stateS, TERMINAL_W.subtract(stateS)).number().intValue() + 1);
  }

  Scalar reward(Tensor state, Tensor action) {
    return KroneckerDelta.of(state, TERMINAL_W);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    // this ensures that staying in the terminal state does not increase the value to infinity
    if (state.equals(TERMINAL_W))
      return RealScalar.ONE;
    // ---
    final Scalar stateS = state.Get();
    Tensor values = Tensors.empty();
    Tensor probs = Tensors.of(P_win, RealScalar.ONE.subtract(P_win));
    // Shangtong Zhang uses
    // headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action]
    // which results in values in the interval [0,1]
    { // win
      Scalar next = stateS.add(action);
      values = values.append(gvalues.Get(statesIndex.of(next)));
      // values = values.append(reward(next, null).add(gvalues.Get(statesIndex.of(next))));
    }
    { // lose
      Scalar next = stateS.add(action.negate());
      values = values.append(gvalues.Get(statesIndex.of(next)));
      // values = values.append(reward(next, null).add(gvalues.Get(statesIndex.of(next))));
    }
    return probs.dot(values).Get();
  }
}
