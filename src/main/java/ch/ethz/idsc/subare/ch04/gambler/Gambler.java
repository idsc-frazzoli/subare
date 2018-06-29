// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Min;

/** Example 4.3 p.90: Gambler's problem
 * an action defines the amount of coins to bet
 * the action has to be non-zero unless the capital == 0
 * or the terminal cash has been reached
 * 
 * [no further references are provided in the book] */
public class Gambler implements StandardModel, MonteCarloInterface {
  private final Tensor states;
  final Scalar TERMINAL_W;
  private final Scalar P_win;
  private final Distribution distribution;

  public static Gambler createDefault() {
    return new Gambler(100, RationalScalar.of(4, 10));
  }

  /** @param max stake
   * @param P_win probabilty of winning a coin toss */
  public Gambler(int max, Scalar P_win) {
    states = Range.of(0, max + 1).unmodifiable();
    TERMINAL_W = (Scalar) Last.of(states);
    this.P_win = P_win;
    distribution = BernoulliDistribution.of(P_win);
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
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) { // non-deterministic
    if (RandomVariate.of(distribution).equals(RealScalar.ONE)) // win
      return state.add(action);
    return state.subtract(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) { // deterministic
    return isTerminal(state) ? RealScalar.ZERO : KroneckerDelta.of(next, TERMINAL_W);
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return states.extract(1, states.length() - 1);
  }

  @Override // from TerminalInterface
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
