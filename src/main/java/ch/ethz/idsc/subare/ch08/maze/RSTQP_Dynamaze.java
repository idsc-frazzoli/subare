// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

/**  */
enum RSTQP_Dynamaze {
  ;
  public static void main(String[] args) throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        dynamaze, qsa, ConstantLearningRate.of(RealScalar.ONE));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_qsa_rstqp.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 50;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(dynamaze, index, ref, qsa);
        TabularSteps.batch(dynamaze, dynamaze, rstqp);
        animationWriter.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
        if (infoline.isLossfree())
          break;
      }
    }
  }
}
