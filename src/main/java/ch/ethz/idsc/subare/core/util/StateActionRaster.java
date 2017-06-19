// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Point;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StateActionRaster extends Raster {
  /** @param state
   * @param action
   * @return point with x, y as coordinates of state-action pair in raster,
   * or null if state-action pair does not have a position in the raster */
  Point point(Tensor state, Tensor action);

  // TODO functions below are EXPERIMENTAL document
  Scalar scaleLoss();

  Scalar scaleQdelta();

  int joinAlongDimension();

  int magify();
}
