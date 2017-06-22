// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Dimensions;

class WireloopRaster implements StateRaster {
  private final Wireloop wireloop;

  public WireloopRaster(Wireloop wireloop) {
    this.wireloop = wireloop;
  }

  @Override
  public DiscreteModel discreteModel() {
    return wireloop;
  }

  @Override
  public Dimension dimensionStateRaster() {
    List<Integer> dimensions = Dimensions.of(wireloop.image());
    return new Dimension(dimensions.get(0), dimensions.get(1));
  }

  @Override
  public Point point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(100.);
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
  public int magify() {
    return 2;
  }
}
