// code by jph
package ch.ethz.idsc.subare.util;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.io.ResourceData;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.opt.LinearInterpolation;

public enum Colorscheme {
  ;
  private static final Interpolation CLASSIC = loadClassic();

  private static Interpolation loadClassic() {
    try {
      return LinearInterpolation.of(ResourceData.of("/colorscheme/classic.csv"));
    } catch (IOException exception) {
      exception.printStackTrace();
    }
    return null;
  }

  private static final Interpolation PARULA = load("parula");

  private static Interpolation load(String string) {
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
