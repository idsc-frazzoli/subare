// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.StateRasters;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Import;

enum WireloopHelper {
  ;
  // ---
  static Wireloop create(String trackName, Function<Tensor, Scalar> function) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Wireloop(image, function);
  }

  static DiscreteQsa getOptimalQsa(Wireloop wireloop) {
    ActionValueIteration avi = new ActionValueIteration(wireloop);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  public static Tensor render(Wireloop wireloop, DiscreteQsa qsa) {
    return render(wireloop, DiscreteUtils.createVs(wireloop, qsa));
  }

  public static Tensor render(Wireloop wireloop, DiscreteVs vs) {
    Tensor tensor = wireloop.image();
    return ImageResize.of(StateRasters.render( //
        new WireloopRaster(wireloop), TensorValuesUtils.rescaled(vs)), 128 / tensor.length());
  }

  /***************************************************/
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
