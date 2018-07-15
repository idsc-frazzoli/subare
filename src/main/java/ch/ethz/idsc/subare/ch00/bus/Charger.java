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

class Charger extends DeterministicStandardModel implements TerminalInterface {
  private final TripProfile tripProfile;
  private final Clip clipCapacity;
  private final Tensor states;
  private final Tensor actions = Range.of(0, 5).unmodifiable();
  public final Dimension dimension;

  public Charger(TripProfile tripProfile, int capacity) {
    this.tripProfile = tripProfile;
    states = Flatten.of(Array.of(Tensors::vector, tripProfile.length(), capacity), 1).unmodifiable();
    clipCapacity = Clip.function(0, capacity - 1);
    dimension = new Dimension(tripProfile.length(), capacity);
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
    Tensor next = state.copy();
    next.set(Increment.ONE, 0);
    Scalar drawn = tripProfile.unitsDrawn(time);
    next.set(capacity -> capacity.add(action).subtract(drawn), 1);
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
    Scalar total = tripProfile.costPerUnit(time).multiply(action.Get()).negate();
    if (capacity == 0)
      total = total.add(RealScalar.of(-20)); // TODO possibly make terminal
    return total;
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.Get(0).number().intValue() == tripProfile.length() - 1;
  }
}
