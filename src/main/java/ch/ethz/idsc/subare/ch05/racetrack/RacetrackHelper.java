// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum RacetrackHelper {
  ;
  static Racetrack create(String trackName, int maxSpeed) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Racetrack(image, maxSpeed);
  }

  private static final Tensor BASE = Tensors.vector(255);

  static Tensor render(Racetrack racetrack, DiscreteQsa qsa, Tensor speed, Tensor action) {
    Interpolation colorscheme = Colorscheme.classic();
    Tensor image = racetrack.image().multiply(RealScalar.ZERO);
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).flatten(0));
    for (Tensor state : racetrack.states())
      if (state.extract(2, 4).equals(speed)) {
        Index index = Index.build(racetrack.actions(state));
        if (index.containsKey(action))
          try {
            Scalar sca = scaled.value(state, action);
            int px = state.Get(0).number().intValue();
            int py = state.Get(1).number().intValue();
            image.set(colorscheme.get(BASE.multiply(sca)), px, py);
          } catch (Exception exception) {
            // ---
          }
      }
    return ImageResize.of(image, 8);
  }
}
