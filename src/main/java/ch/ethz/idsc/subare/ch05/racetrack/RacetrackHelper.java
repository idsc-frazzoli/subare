// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.Import;

enum RacetrackHelper {
  ;
  static Racetrack create(String trackName, int maxSpeed) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Racetrack(image, maxSpeed);
  }

  static Tensor render(Racetrack racetrack, DiscreteQsa qsa, Tensor speed, Tensor action) {
    Tensor tensor = racetrack.image().get(Tensor.ALL, Tensor.ALL, 0).map(scalar -> DoubleScalar.INDETERMINATE);
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).flatten(0));
    for (Tensor state : racetrack.states())
      if (state.length() == 4 && state.extract(2, 4).equals(speed)) {
        Index index = Index.build(racetrack.actions(state));
        if (index.containsKey(action))
          try {
            Scalar sca = scaled.value(state, action);
            int px = state.Get(0).number().intValue();
            int py = state.Get(1).number().intValue();
            tensor.set(sca, px, py);
          } catch (Exception exception) {
            // ---
          }
      }
    Tensor image = ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 8);
  }
}
