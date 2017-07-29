// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.io.ResourceData;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.opt.LinearInterpolation;

public enum Colorscheme {
  ;
  @Deprecated
  public static final Interpolation CLASSIC = LinearInterpolation.of(ResourceData.of("/colorscheme/classic.csv"));
}
