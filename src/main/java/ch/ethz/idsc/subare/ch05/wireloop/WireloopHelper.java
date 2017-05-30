// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.red.Max;

enum WireloopHelper {
  ;
  // ---
  static Wireloop create(String trackName, Function<Tensor, Scalar> function) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Wireloop(image, function);
  }

  private static final Tensor BASE = Tensors.vector(255);

  public static Tensor render(Wireloop wireloop, DiscreteQsa qsa) {
    Interpolation colorscheme = Colorscheme.classic();
    Tensor tensor = wireloop.image();
    // final int length = wireloop.states().length();
    // final int sizea = (length + 1) / 2;
    // final int offset = (length - 1) / 2;
    // final Tensor tensor = Array.zeros(length, sizea, 4);
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).flatten(0));
    for (Tensor state : wireloop.states()) {
      Scalar max = RealScalar.of(-1e10);
      for (Tensor action : wireloop.actions(state)) {
        Scalar sca = scaled.value(state, action);
        max = Max.of(max, sca);
      }
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      tensor.set(colorscheme.get(BASE.multiply(max)), x, y);
    }
    return ImageResize.of(tensor, 3);
  }
}
