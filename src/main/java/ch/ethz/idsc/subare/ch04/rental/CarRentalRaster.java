// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateActionRaster;
import ch.ethz.idsc.tensor.Tensor;

class CarRentalRaster implements StateActionRaster {
  private final CarRental carRental;

  public CarRentalRaster(CarRental carRental) {
    this.carRental = carRental;
  }

  @Override
  public DiscreteModel discreteModel() {
    return carRental;
  }

  @Override
  public Dimension dimension() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Point point(Tensor state, Tensor action) {
    // TODO Auto-generated method stub
    return null;
  }
}
