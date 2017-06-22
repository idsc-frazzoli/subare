// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.Arrays;

import ch.ethz.idsc.subare.core.util.gfx.DefaultStateRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

class CliffwalkStateRaster extends DefaultStateRaster {
  public CliffwalkStateRaster(Cliffwalk cliffwalk) {
    super(cliffwalk, Arrays.asList(cliffwalk.NX, cliffwalk.NY));
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
  public int magify() {
    return 5;
  }
}
