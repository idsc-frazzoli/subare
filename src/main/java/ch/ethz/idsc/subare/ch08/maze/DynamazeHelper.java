// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.io.File;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.StateRasters;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.io.Import;

enum DynamazeHelper {
  ;
  public static Dynamaze create(String name) throws Exception {
    String path = "".getClass().getResource("/ch08/" + name + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Dynamaze(image);
  }

  static DiscreteQsa getOptimalQsa(Dynamaze dynamaze) {
    return ActionValueIterations.solve(dynamaze, DecimalScalar.of(.0001));
  }

  private static final int MAGNIFY = 5;

  static Tensor render(Dynamaze dynamaze, DiscreteVs vs) {
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    return ImageResize.of(StateRasters.render(new DynamazeRaster(dynamaze), scaled), MAGNIFY);
  }
}
