// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.api.TensorScalarFunction;

/** Example 5.2 p.95: Soap Bubble
 * a classical Dirichlet problem
 * 
 * References:
 * Kakutani (1945)
 * Hersh, and Griego (1969)
 * Doyle, and Snell (1984) */
class Wireloop extends DeterministicStandardModel implements MonteCarloInterface {
  static final Tensor WHITE = Tensors.vector(255, 255, 255, 255);
  static final Tensor GREEN = Tensors.vector(0, 255, 0, 255);
  // ---
  static final Tensor ACTIONS = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  static final Tensor ACTIONS_TERMINAL = Array.zeros(1, 2).unmodifiable();
  // ---
  private final Tensor image;
  private final TensorScalarFunction function;
  private final WireloopReward wireloopReward;
  private final Tensor states = Tensors.empty();
  private final Set<Tensor> startStates = new HashSet<>();
  private final Set<Tensor> endStates = new HashSet<>();
  // private final StateActionMap stateActionMap = StateActionMap.empty();

  Wireloop(Tensor image, TensorScalarFunction function, WireloopReward wireloopReward) {
    this.image = Objects.requireNonNull(image);
    this.function = function;
    this.wireloopReward = wireloopReward;
    // System.out.println(Dimensions.of(image));
    List<Integer> dims = Dimensions.of(image);
    for (int y = 0; y < dims.get(0); ++y)
      for (int x = 0; x < dims.get(1); ++x) {
        final Tensor rgba = image.get(y, x);
        if (rgba.equals(GREEN)) {
          Tensor row = Tensors.vector(x, y);
          startStates.add(row);
          states.append(row);
        } else //
        if (rgba.equals(WHITE)) {
          Tensor row = Tensors.vector(x, y);
          endStates.add(row);
          states.append(row);
        }
      }
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return isTerminal(state) //
        ? ACTIONS_TERMINAL // list of actions {...} with only single action = {0, 0}
        : ACTIONS;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    if (startStates.contains(state) && endStates.contains(next))
      return function.apply(next);
    return wireloopReward.reward(state, action, next);
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensor.of(startStates.stream());
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return endStates.contains(state);
  }

  /**************************************************/
  public Tensor image() {
    return image.copy();
  }
}
