// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.color.Colorscheme;
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
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).flatten(0));
    for (Tensor state : wireloop.states()) {
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      Scalar max = wireloop.actions(state).flatten(0) //
          .map(action -> scaled.value(state, action)) //
          .reduce(Max::of).get();
      tensor.set(colorscheme.get(BASE.multiply(max)), x, y);
    }
    return ImageResize.of(tensor, 3);
  }

  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
