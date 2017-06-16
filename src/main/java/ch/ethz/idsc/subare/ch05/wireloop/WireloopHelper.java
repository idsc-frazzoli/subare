// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.List;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.core.util.StateRasters;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.io.Import;

enum WireloopHelper {
  ;
  private static final int MAGNIFY = 3;

  static Wireloop create(String trackName, Function<Tensor, Scalar> function, Scalar stepCost) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Wireloop(image, function, stepCost);
  }

  static Wireloop create(String trackName, Function<Tensor, Scalar> function) throws Exception {
    return create(trackName, function, RealScalar.ZERO);
  }

  static DiscreteQsa getOptimalQsa(Wireloop wireloop) {
    return ActionValueIterations.solve(wireloop, DecimalScalar.of(.0001));
  }

  public static Tensor render(Wireloop wireloop, DiscreteQsa qsa) {
    return render(wireloop, DiscreteUtils.createVs(wireloop, qsa));
  }

  public static Tensor render(Wireloop wireloop, DiscreteQsa ref, DiscreteQsa qsa) {
    Tensor im1 = render(wireloop, DiscreteUtils.createVs(wireloop, qsa));
    List<Integer> dimensions = Dimensions.of(im1);
    dimensions.set(0, MAGNIFY);
    DiscreteVs vs = Loss.perState(wireloop, ref, qsa);
    Tensor im2 = render(wireloop, vs);
    return Join.of(im1, Array.zeros(dimensions), im2);
  }

  public static Tensor render(Wireloop wireloop, DiscreteVs vs) {
    return ImageResize.of(
        StateRasters.render( //
            new WireloopRaster(wireloop), TensorValuesUtils.rescaled(vs)), //
        MAGNIFY //
    // 128 / tensor.length() //
    );
  }

  /***************************************************/
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
