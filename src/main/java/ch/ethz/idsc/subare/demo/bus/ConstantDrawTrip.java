// code by jph
package ch.ethz.idsc.subare.demo.bus;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Range;

class ConstantDrawTrip implements TripProfile {
  private final int length;
  private final Tensor costPerUnit;
  private final Scalar draw;

  public ConstantDrawTrip(int length, int amount) {
    this.length = length;
    costPerUnit = Range.of(0, length).map(new Sawtooth(3)); // .map(Increment.ONE);
    draw = RealScalar.of(amount);
  }

  @Override
  public int length() {
    return length;
  }

  @Override
  public Scalar costPerUnit(int index) {
    return costPerUnit.Get(index);
  }

  @Override
  public Scalar unitsDrawn(int time) {
    return draw;
  }
}
