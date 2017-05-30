// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.ActionValueInterface;
import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.subare.core.RewardInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

class Wireloop implements StandardModel, MoveInterface, RewardInterface, ActionValueInterface {
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
  final Function<Tensor, Scalar> function;
  final Tensor states = Tensors.empty();
  final Set<Tensor> startStates = new HashSet<>();
  final Set<Tensor> endStates = new HashSet<>();

  Wireloop(Tensor image, Function<Tensor, Scalar> function) {
    this.image = image;
    this.function = function;
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
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
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
  public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    throw new RuntimeException();
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (startStates.contains(state) && endStates.contains(next)) {
      return function.apply(next);
    }
    return ZeroScalar.get();
  }

  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    return reward(state, action, move(state, action));
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(move(state, action), next);
  }

  public Tensor image() {
    return image.copy();
  }
}
