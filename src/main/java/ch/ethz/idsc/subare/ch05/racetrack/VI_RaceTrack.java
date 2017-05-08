// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.ExtractPrimitives;
import ch.ethz.idsc.tensor.io.Import;

class VI_RaceTrack {
  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track1.png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    Racetrack racetrack = new Racetrack(image, 5);
    ValueIteration vi = new ValueIteration(racetrack, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(10), 5);
    final Tensor values = vi.vs().values();
    System.out.println("iterations=" + vi.iterations());
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobableGreedy(racetrack, values);
    int k = 0;
    for (Tensor start : racetrack.statesStart) {
      Tensor copy = image.copy();
      MonteCarloEpisode mce = new MonteCarloEpisode( //
          racetrack, policyInterface, start);
      while (mce.hasNext()) {
        StepInterface stepInterface = mce.step();
        {
          Tensor state = stepInterface.prevState();
          int[] index = ExtractPrimitives.toArrayInt(state);
          copy.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
        {
          Tensor state = stepInterface.nextState();
          int[] index = ExtractPrimitives.toArrayInt(state);
          copy.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
      }
      Export.of( //
          new File("/home/datahaki/Pictures/racetrack", String.format("track1_%02d.png", k)), //
          ImageResize.of(copy, 8));
      ++k;
    }
  }
}
