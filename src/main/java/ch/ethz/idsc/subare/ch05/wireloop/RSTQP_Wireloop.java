// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;

/** Example 4.1, p.82 */
class RSTQP_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        wireloop, qsa, ConstantLearningRate.of(RealScalar.ONE));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L_qsa_rstqp.gif"), 250);
    int batches = 50;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
      TabularSteps.batch(wireloop, wireloop, rstqp);
      gsw.append(WireloopHelper.render(wireloopRaster, ref, qsa));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
  }
}
