// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Scalar;

interface Raster {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return loss function scale for visualization */
  Scalar scaleLoss();

  /** @return q function error scale for visualization */
  Scalar scaleQdelta();

  /** @return either 0 or 1 as dimension to join q function, loss, etc. */
  int joinAlongDimension();

  /** @return magnification */
  int magnify();
}
