// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.Random;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
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
class Gambler implements StandardModel, //
    MonteCarloInterface, EpisodeSupplier, ActionValueInterface {
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

  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(ZeroScalar.get());
    // here we deviate from the book and the code by STZ:
    // we require that the bet=action is non-zero,
    // if the state is non-terminal, 0 < cash < 100.
    // otherwise the player can stall (the iteration) forever.
    Scalar stateS = state.Get();
    return Range.of(1, Min.of(stateS, TERMINAL_W.subtract(stateS)).number().intValue() + 1);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
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
      values = values.append(gvalues.value(next));
      // values = values.append(reward(next, null).add(gvalues.Get(statesIndex.of(next))));
    }
    { // lose
      Scalar next = stateS.add(action.negate());
      values = values.append(gvalues.value(next));
      // values = values.append(reward(next, null).add(gvalues.Get(statesIndex.of(next))));
    }
    return probs.dot(values).Get();
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
    return KroneckerDelta.of(next, TERMINAL_W);
  }

  /**************************************************/
  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    Tensor start = states.get(random.nextInt(states.length() - 2) + 1);
    if (isTerminal(start))
      throw new RuntimeException();
    return new MonteCarloEpisode(this, policyInterface, start);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.equals(ZeroScalar.get()) || state.equals(TERMINAL_W);
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return ZeroScalar.get();
    if (state.add(action).equals(TERMINAL_W))
      return P_win; // P_win * 1
    return ZeroScalar.get();
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    if (isTerminal(state))
      return Tensors.of(state); // next \in {state}
    return Tensors.of( //
        state.add(action), // P_win
        state.subtract(action)); // 1 - P_win
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

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }
}
