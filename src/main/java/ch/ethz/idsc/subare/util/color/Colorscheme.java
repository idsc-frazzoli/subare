// code by jph
package ch.ethz.idsc.subare.util.color;

import java.io.File;

import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.opt.LinearInterpolation;

public class Colorscheme {
  public static Interpolation classic() throws Exception {
    return LinearInterpolation.of( //
        Import.of(new File("".getClass().getResource("/util/colorscheme/classic.csv").getPath())));
  }

  public static void main(String[] args) throws Exception {
    // Export.of(UserHome.file("classic.csv"), classic());
  }
}
