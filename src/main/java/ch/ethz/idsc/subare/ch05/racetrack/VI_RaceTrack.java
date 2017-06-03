// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.util.LinkedList;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.ExtractPrimitives;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class VI_RaceTrack {
  public static void main(String[] args) throws Exception {
    final String name = "track2";
    Racetrack racetrack = RacetrackHelper.create(name, 5);
    ValueIteration vi = new ValueIteration(racetrack);
    vi.untilBelow(DecimalScalar.of(10), 5);
    System.out.println("iterations=" + vi.iterations());
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobableGreedy(racetrack, vi.vs());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + ".gif"), 400);
    for (Tensor start : racetrack.statesStart) {
      Tensor image = racetrack.image();
      MonteCarloEpisode mce = new MonteCarloEpisode( //
          racetrack, policyInterface, start, new LinkedList<>());
      while (mce.hasNext()) {
        StepInterface stepInterface = mce.step();
        {
          Tensor state = stepInterface.prevState();
          int[] index = ExtractPrimitives.toArrayInt(state);
          image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
        {
          Tensor state = stepInterface.nextState();
          int[] index = ExtractPrimitives.toArrayInt(state);
          image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
      }
      gsw.append(ImageFormat.of(ImageResize.of(image, 6)));
    }
    gsw.close();
    System.out.println("gif created");
  }
}
