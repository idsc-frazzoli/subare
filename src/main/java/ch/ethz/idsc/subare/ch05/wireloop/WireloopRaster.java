// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.util.DefaultStateRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.alg.Dimensions;

public class WireloopRaster extends DefaultStateRaster {
  public WireloopRaster(Wireloop wireloop) {
    super(wireloop, Dimensions.of(wireloop.image()));
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
