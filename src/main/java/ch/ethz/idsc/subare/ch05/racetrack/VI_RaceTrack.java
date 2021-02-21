// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.util.LinkedList;
import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.Primitives;

/* package */ enum VI_RaceTrack {
  ;
  public static void make(String name, int maxSpeed, File file) throws Exception {
    Racetrack racetrack = RacetrackHelper.create(name, maxSpeed);
    ValueIteration vi = new ValueIteration(racetrack, racetrack);
    vi.untilBelow(RealScalar.of(10), 5);
    System.out.println("iterations=" + vi.iterations());
    Policy policy = PolicyType.GREEDY.bestEquiprobable(racetrack, vi.vs(), null);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(file, 400, TimeUnit.MILLISECONDS)) {
      for (Tensor start : racetrack.statesStart) {
        System.out.println(start);
        Tensor image = racetrack.image();
        MonteCarloEpisode mce = new MonteCarloEpisode( //
            racetrack, policy, start, new LinkedList<>());
        while (mce.hasNext()) {
          StepInterface stepInterface = mce.step();
          {
            Tensor state = stepInterface.prevState();
            int[] index = Primitives.toIntArray(state);
            image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
          }
          {
            Tensor state = stepInterface.nextState();
            int[] index = Primitives.toIntArray(state);
            image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
          }
        }
        animationWriter.write(ImageResize.nearest(image, 6));
      }
    }
    System.out.println("gif created");
  }

  public static void main(String[] args) throws Exception {
    make("track2", 5, HomeDirectory.Pictures(VI_RaceTrack.class.getSimpleName() + ".gif"));
  }
}
