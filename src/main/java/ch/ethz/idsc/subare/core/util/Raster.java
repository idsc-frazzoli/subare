// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Dimension;
import java.io.Serializable;

import ch.ethz.idsc.subare.core.DiscreteModel;

interface Raster extends Serializable {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return dimension of raster */
  Dimension dimension();
}
