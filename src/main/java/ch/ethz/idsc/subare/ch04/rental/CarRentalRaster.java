// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
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
  public Dimension dimensionStateRaster() {
    return new Dimension(carRental.maxCars + 1, carRental.maxCars + 1);
  }

  @Override
  public Point point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.ONE;
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.ONE;
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magnify() {
    return 4;
  }
}
