// code by jph
package ch.ethz.idsc.subare.util;

import java.io.File;

import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.opt.LinearInterpolation;

public enum Colorscheme {
  ;
  private static final Interpolation CLASSIC = of("classic");
  // LinearInterpolation.of(ResourceData.of("/colorscheme/classic.csv"));
  private static final Interpolation PARULA = of("parula");

  private static Interpolation of(String string) {
    try {
      return LinearInterpolation.of( //
          Import.of(new File("".getClass().getResource("/util/colorscheme/" + string + ".csv").getPath())));
    } catch (Exception exception) {
      // ---
    }
    return null;
  }

  public static Interpolation classic() {
    return CLASSIC;
  }

  public static Interpolation parula() {
    return PARULA;
  }
}
