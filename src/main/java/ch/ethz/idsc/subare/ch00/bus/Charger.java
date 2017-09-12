// code by jph
package ch.ethz.idsc.subare.ch00.bus;

import java.awt.Dimension;

import ch.ethz.idsc.subare.core.TerminalInterface;
import ch.ethz.idsc.subare.core.adapter.DeterministicStandardModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Flatten;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Increment;

public class Charger extends DeterministicStandardModel implements TerminalInterface {
  public final int action_length = 4;
  private final int times;
  // public final int Capacity;
  private final Clip clipCapacity;
  private final Tensor states;
  private final Tensor actions = Tensors.vector(i -> Tensors.vector(1, i), action_length).unmodifiable();
  private final Tensor gains;
  private final Tensor draw = Tensors.vector( //
      2, 2, 2, 3, 3, 2, 2, 3, 1, 1, 4, 4, 1, 2, 4, 3, 2, 1, 1, 3, 3, 3, 2, 1, 0, 0, 0);
  public final Dimension dimension;

  public Charger(int times, int capacity) {
    this.times = times;
    states = Flatten.of(Array.of(Tensors::vector, times, capacity), 1).unmodifiable();
    Sawtooth sawtooth = new Sawtooth(3);
    gains = Range.of(0, times).map(sawtooth).map(Increment.ONE).negate();
    clipCapacity = Clip.function(0, capacity - 1);
    dimension = new Dimension(times, capacity);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    final int time = state.Get(0).number().intValue();
    Tensor next = state.add(action);
    next.set(s -> s.subtract(draw.Get(time)), 1);
    next.set(clipCapacity, 1);
    return next;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    final int time = state.Get(0).number().intValue();
    final int capacity = state.Get(1).number().intValue();
    if (isTerminal(next)) {
      if (isTerminal(state))
        return RealScalar.ZERO;
      return RealScalar.of(0 == capacity ? -10 : 0);
    }
    if (capacity == 0)
      return RealScalar.of(-10); // TODO possibly make terminal
    return gains.Get(time).multiply(action.Get(1));
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.Get(0).number().intValue() == times - 1;
  }
}
