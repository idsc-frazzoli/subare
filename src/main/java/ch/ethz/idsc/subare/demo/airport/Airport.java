// code fluric
package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Example 4.3 p.90: Gambler's problem
 * an action defines the amount of coins to bet
 * the action has to be non-zero unless the capital == 0
 * or the terminal cash has been reached
 * 
 * [no further references are provided in the book] */
class Airport implements StandardModel, MonteCarloInterface {
  private static final int LASTT = 3;
  private static final int VEHICLES = 5;
  private final Tensor states;

  /** @param max stake
   * @param P_win probabilty of winning a coin toss */
  public Airport() {
    states = Tensors.empty();
    states.append(Tensors.vector(0, 5, 0));
    for (int t = 1; t <= LASTT; t++) {
      for (int v = 0; v <= VEHICLES; v++) {
        states.append(Tensors.vector(t, v, VEHICLES - v));
      }
    }
  }

  @Override
  public Tensor states() { // done
    return states;
  }

  /** @return possible stakes */
  @Override
  public Tensor actions(Tensor state) { // done
    if (isTerminal(state))
      return Tensors.fromString("{{0}}");
    // return Tensors.of(RealScalar.ZERO);
    // here we deviate from the book and the code by STZ:
    // we require that the bet=action is non-zero,
    // if the state is non-terminal, 0 < cash < 100.
    // otherwise the player can stall (the iteration) forever.
    Tensor actions = Tensors.empty();
    for (int i = -state.Get(2).number().intValue(); i <= state.Get(1).number().intValue(); i++) {
      actions.append(Tensors.vector(1, - i, i));
    }
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) { // done
    if (isTerminal(state)) {
      return state;
    }
    return state.add(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) { // deterministic
    return isTerminal(state) ? RealScalar.ZERO : RealScalar.ZERO; // TODO
  }

  /**************************************************/
  @Override // from MonteCarloInterface // done
  public Tensor startStates() {
    return states.extract(0, 1);
  }

  @Override // from TerminalInterface, done
  public boolean isTerminal(Tensor state) {
    return state.get(0).equals(RealScalar.of(LASTT));
  }

  /**************************************************/
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    if (state.Get(0).equals(RealScalar.of(2)) && state.Get(2).equals(RealScalar.of(3)))
      return RealScalar.ONE;
    return RealScalar.ZERO; // TODO
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) { // done
    if (isTerminal(state))
      return Tensors.of(state); // next \in {state}
    return Tensors.of( //
        state.add(action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ONE;
    if (state.add(action).equals(next))
      return RealScalar.ONE;
    return RealScalar.ZERO;
  }
}
