// code by jph
package ch.ethz.idsc.subare.demo.bus;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;

/* package */ class ChargerRaster implements StateActionRaster {
  private final Charger charger;
  private final int SPACE_Y;

  public ChargerRaster(Charger charger) {
    this.charger = charger;
    SPACE_Y = charger.dimension.height + 1;
  }

  @Override
  public DiscreteModel discreteModel() {
    return charger;
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
    return 1;
  }

  @Override
  public int magnify() {
    return 6;
  }

  @Override
  public Dimension dimensionStateActionRaster() {
    Dimension dimension = new Dimension();
    dimension.width = charger.dimension.width;
    dimension.height = charger.actions(null).length() * SPACE_Y - 1;
    return dimension;
  }

  @Override
  public Point point(Tensor state, Tensor action) {
    int x = Scalars.intValueExact(state.Get(0));
    int y = SPACE_Y - Scalars.intValueExact(state.Get(1)) - 2;
    int load = 4 - Scalars.intValueExact((Scalar) action);
    load *= SPACE_Y;
    return new Point(x, y + load);
  }
}
