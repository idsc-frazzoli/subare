// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.tensor.Tensor;

public interface StateRaster extends Raster {
  /** @return dimension of raster */
  Dimension dimensionStateRaster();

  /** @param state
   * @return point with x, y as coordinates of state in raster,
   * or null if state does not have a position in the raster */
  Point point(Tensor state);
}
