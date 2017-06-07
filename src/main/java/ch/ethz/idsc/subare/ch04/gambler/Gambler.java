// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.Random;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Min;

/** Gambler's problem:
 * an action defines the amount of coins to bet
 * the action has to be non-zero unless the capital == 0
 * or the terminal cash has been reached */
class Gambler implements StandardModel, //
    MonteCarloInterface, ActionValueInterface {
  private final Tensor states;
  final Scalar TERMINAL_W;
  final Scalar P_win;
  Random random = new Random();

  public static Gambler createDefault() {
    return new Gambler(100, RationalScalar.of(4, 10));
  }

  /** @param max stake
   * @param P_win probabilty of winning a coin toss */
  public Gambler(int max, Scalar P_win) {
    states = Range.of(0, max + 1).unmodifiable();
    TERMINAL_W = (Scalar) Last.of(states);
    this.P_win = P_win;
  }

  @Override
  public Tensor states() {
    return states;
  }

  /** @return possible stakes */
  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(RealScalar.ZERO);
    // here we deviate from the book and the code by STZ:
    // we require that the bet=action is non-zero,
    // if the state is non-terminal, 0 < cash < 100.
    // otherwise the player can stall (the iteration) forever.
    Scalar stateS = state.Get();
    return Range.of(1, Min.of(stateS, TERMINAL_W.subtract(stateS)).number().intValue() + 1);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    if (isTerminal(state))
      return gvalues.value(state);
    // ---
    final Scalar stateS = state.Get(); // current total available to gambler
    Tensor values = Tensors.empty();
    Tensor probs = Tensors.of(P_win, RealScalar.ONE.subtract(P_win));
    { // win
      Scalar next = stateS.add(action); // stake is added to current total
      values = values.append(gvalues.value(next));
    }
    { // lose
      Scalar next = stateS.subtract(action); // stake is subtracted from current total
      values = values.append(gvalues.value(next));
    }
    return probs.dot(values).Get();
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (Scalars.lessThan(DoubleScalar.of(random.nextDouble()), P_win))
      return state.add(action);
    return state.subtract(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return isTerminal(state) ? RealScalar.ZERO : KroneckerDelta.of(next, TERMINAL_W);
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return states.extract(1, states.length() - 1);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(RealScalar.ZERO) || state.equals(TERMINAL_W);
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    return KroneckerDelta.of(state.add(action), TERMINAL_W).multiply(P_win); // P_win * 1, or 0
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    if (isTerminal(state))
      return Tensors.of(state); // next \in {state}
    return Tensors.of( //
        state.add(action), // with probability P_win
        state.subtract(action)); // with probability 1 - P_win
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ONE;
    if (state.add(action).equals(next))
      return P_win;
    if (state.subtract(action).equals(next))
      return RealScalar.ONE.subtract(P_win);
    throw new RuntimeException();
  }
}
