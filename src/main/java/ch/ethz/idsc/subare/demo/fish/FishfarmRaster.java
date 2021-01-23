// code by jph
package ch.ethz.idsc.subare.demo.fish;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;

class FishfarmRaster implements StateRaster {
  private final Fishfarm fishfarm;

  public FishfarmRaster(Fishfarm fishfarm) {
    this.fishfarm = fishfarm;
  }

  @Override
  public DiscreteModel discreteModel() {
    return fishfarm;
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(1.0);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(1.0);
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magnify() {
    return 5;
  }

  @Override
  public Dimension dimensionStateRaster() {
    return new Dimension(fishfarm.period + 1, fishfarm.max_fish + 1);
  }

  @Override
  public Point point(Tensor state) {
    int sx = Scalars.intValueExact(state.Get(0));
    int sy = fishfarm.max_fish - Scalars.intValueExact(state.Get(1));
    return new Point(sx, sy);
  }
}
