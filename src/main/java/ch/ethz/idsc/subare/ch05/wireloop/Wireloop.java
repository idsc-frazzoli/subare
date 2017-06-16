// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.subare.core.util.StateActionMap;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;

/** Example 5.2 p.103: Soap Bubble
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
  static final Tensor actions = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  // ---
  private final Tensor image;
  private final Function<Tensor, Scalar> function;
  private final Scalar stepCost;
  private final Tensor states = Tensors.empty();
  private final Set<Tensor> startStates = new HashSet<>();
  private final Set<Tensor> endStates = new HashSet<>();
  private final StateActionMap stateActionMap = StateActionMap.empty();

  Wireloop(Tensor image, Function<Tensor, Scalar> function, Scalar stepCost) {
    this.image = image;
    this.function = function;
    this.stepCost = stepCost;
    System.out.println(Dimensions.of(image));
    List<Integer> dims = Dimensions.of(image);
    for (int x = 0; x < dims.get(0); ++x)
      for (int y = 0; y < dims.get(1); ++y) {
        Tensor rgba = image.get(x, y);
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
    for (Tensor state : states)
      stateActionMap.put(state, _actions(state));
  }

  @Override
  public Tensor states() {
    return states;
  }

  private Tensor _actions(Tensor state) {
    if (startStates.contains(state)) {
      Tensor tensor = Tensors.empty();
      for (Tensor action : actions) {
        Tensor probe = state.add(action);
        if (startStates.contains(probe) || endStates.contains(probe))
          tensor.append(action);
      }
      return tensor;
    }
    return Array.zeros(1, 2);
  }

  @Override
  public Tensor actions(Tensor state) {
    return stateActionMap.actions(state);
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
    return stepCost;
  }

  /**************************************************/
  @Override
  public Tensor startStates() {
    return Tensor.of(startStates.stream());
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return endStates.contains(state);
  }

  /**************************************************/
  public Tensor image() {
    return image.copy();
  }
}
