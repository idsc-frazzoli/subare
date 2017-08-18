// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.io.File;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Import;

enum DynamazeHelper {
  ;
  private static final Tensor STARTS = Tensors.matrixInt(new int[][] { //
      { 31, 15 }, { 9, 15 }, { 18, 12 } });

  @Deprecated
  public static StateRaster createRaster(Dynamaze dynamaze) {
    return new DynamazeRaster(dynamaze);
  }

  public static Dynamaze original(String name) throws Exception {
    return fromImage(load(name));
  }

  public static Dynamaze create5(int starts) throws Exception {
    Tensor image = load("maze5");
    for (int count = 0; count < starts; ++count) {
      Tensor vec = STARTS.get(count);
      image.set(Dynamaze.GREEN, //
          vec.Get(0).number().intValue(), //
          vec.Get(1).number().intValue());
    }
    return fromImage(image);
  }

  private static Dynamaze fromImage(Tensor image) {
    return new Dynamaze(image.unmodifiable());
  }

  /* package */ static Tensor load(String name) throws Exception {
    String path = "".getClass().getResource("/ch08/" + name + ".png").getPath();
    return Import.of(new File(path));
  }

  static DiscreteQsa getOptimalQsa(Dynamaze dynamaze) {
    return ActionValueIterations.solve(dynamaze, DecimalScalar.of(.0000001));
  }
}
