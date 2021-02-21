// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.ResourceData;

public enum DynamazeHelper {
  ;
  private static final Tensor STARTS_5 = Tensors.matrixInt(new int[][] { //
      { 31, 15 }, { 9, 15 }, { 18, 12 } });

  @Deprecated
  public static StateRaster createRaster(Dynamaze dynamaze) {
    return new DynamazeRaster(dynamaze);
  }

  /** @param name, for instance "maze2"
   * @return */
  public static Dynamaze original(String name) {
    return fromImage(load(name));
  }

  public static Dynamaze create5(int starts) {
    Tensor image = load("maze5");
    for (int count = 0; count < starts; ++count) {
      Tensor vec = STARTS_5.get(count);
      image.set(Dynamaze.GREEN, //
          Scalars.intValueExact(vec.Get(0)), //
          Scalars.intValueExact(vec.Get(1)));
    }
    return fromImage(image);
  }

  private static Dynamaze fromImage(Tensor image) {
    return new Dynamaze(image.unmodifiable());
  }

  /* package */ static Tensor load(String name) {
    return ResourceData.of("/ch08/" + name + ".png");
  }

  static DiscreteQsa getOptimalQsa(Dynamaze dynamaze) {
    return ActionValueIterations.solve(dynamaze, RealScalar.of(.0000001));
  }
}
