// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.util.Arrays;

import ch.ethz.idsc.subare.core.util.gfx.DefaultStateRaster;
import ch.ethz.idsc.tensor.Scalar;

public class CarRentalRaster extends DefaultStateRaster {
  public CarRentalRaster(CarRental carRental) {
    super(carRental, Arrays.asList(carRental.maxCars + 1, carRental.maxCars + 1));
  }

  @Override
  public Scalar scaleLoss() {
    return null; // TODO
  }

  @Override
  public Scalar scaleQdelta() {
    return null; // TODO
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magify() {
    return 4;
  }
}
