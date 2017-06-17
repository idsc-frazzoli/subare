// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;

class CarRentalRaster implements StateRaster {
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
    return new Dimension(carRental.maxCars + 1, carRental.maxCars + 1);
  }

  @Override
  public Point point(Tensor state) {
    int x = state.Get(0).number().intValue();
    int y = state.Get(1).number().intValue();
    return new Point(x, y);
  }
}
