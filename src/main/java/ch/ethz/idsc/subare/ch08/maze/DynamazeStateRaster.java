// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.util.DefaultStateRaster;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.alg.Dimensions;

public class DynamazeStateRaster extends DefaultStateRaster {
  public DynamazeStateRaster(Dynamaze dynamaze) {
    super(dynamaze, Dimensions.of(dynamaze.image()));
  }

  @Override
  public Scalar scaleLoss() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar scaleQdelta() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public int joinAlongDimension() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public int magify() {
    return 4;
  }
}
