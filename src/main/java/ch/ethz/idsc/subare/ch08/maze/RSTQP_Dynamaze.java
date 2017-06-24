// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;

/**  */
class RSTQP_Dynamaze {
  public static void main(String[] args) throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        dynamaze, qsa, ConstantLearningRate.of(RealScalar.ONE));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "_qsa_rstqp.gif"), 250);
    int batches = 50;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(dynamaze, index, ref, qsa);
      TabularSteps.batch(dynamaze, dynamaze, rstqp);
      gsw.append(StateRasters.vs_rescale(dynamazeRaster, qsa));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
  }
}
