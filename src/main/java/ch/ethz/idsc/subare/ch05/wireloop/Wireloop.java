// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.DeterministicStandardModel;
import ch.ethz.idsc.subare.core.util.StateActionMap;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;

class Wireloop extends DeterministicStandardModel implements //
    MonteCarloInterface, EpisodeSupplier {
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
  StateActionMap stateActionMap = StateActionMap.empty();

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
  public boolean isTerminal(Tensor state) {
    return endStates.contains(state);
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
    return RealScalar.ZERO;
  }

  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    List<Tensor> starts = startStates.stream().collect(Collectors.toList());
    Collections.shuffle(starts);
    Tensor start = starts.get(0);
    return new MonteCarloEpisode(this, policyInterface, start);
  }

  public Tensor image() {
    return image.copy();
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }
}
