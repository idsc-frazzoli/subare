// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;

/** Example 8.1 p.172: Dyna Maze */
class Dynamaze extends DeterministicStandardModel implements MonteCarloInterface {
  static final Tensor WHITE = Tensors.vector(255, 255, 255, 255);
  static final Tensor RED = Tensors.vector(255, 0, 0, 255);
  static final Tensor GREEN = Tensors.vector(0, 255, 0, 255);
  static final Tensor BLACK = Tensors.vector(0, 0, 0, 255);
  static final Scalar MINUS_ONE = RealScalar.ONE.negate();
  static final Tensor ACTIONS = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  static final Tensor ACTIONS_TERMINAL = Array.zeros(1, 2).unmodifiable();
  // ---
  private final Tensor image;
  private final Tensor states = Tensors.empty();
  private final Index statesIndex;
  private final Tensor startStates = Tensors.empty();
  private final Index terminalIndex;

  Dynamaze(Tensor image) {
    this.image = image;
    List<Integer> list = Dimensions.of(image);
    Tensor terminal = Tensors.empty();
    for (int x = 0; x < list.get(1); ++x)
      for (int y = 0; y < list.get(0); ++y) {
        Tensor color = image.get(y, x);
        if (!color.equals(WHITE)) {
          Tensor state = Tensors.vector(x, y);
          states.append(state);
          if (color.equals(GREEN))
            startStates.append(state);
          else //
          if (color.equals(RED))
            terminal.append(state);
        }
      }
    statesIndex = Index.build(states);
    terminalIndex = Index.build(terminal);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return isTerminal(state) ? ACTIONS_TERMINAL : ACTIONS;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.of(0.95); // TODO check how this behaves with 19/20
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    Tensor next = state.add(action);
    return statesIndex.containsKey(next) ? next : state;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return !isTerminal(state) && isTerminal(next) ? RealScalar.ONE : RealScalar.ZERO;
  }

  /**************************************************/
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return terminalIndex.containsKey(state);
  }

  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return startStates;
  }

  /**************************************************/
  public Tensor image() {
    return image.copy();
  }

  public Index terminalIndex() {
    return terminalIndex;
  }
}
