// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 4.1, p.82 */
class RSTQP_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x, wireloopReward);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        wireloop, qsa, ConstantLearningRate.of(RealScalar.ONE));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L_qsa_rstqp.gif"), 250);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar q_delta = DiscreteValueFunctions.distance(qsa, ref);
      Scalar loss = Loss.accumulation(wireloop, ref, qsa);
      System.out.println(index + " " + q_delta.map(Round._1) + " " + loss);
      TabularSteps.batch(wireloop, wireloop, rstqp);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, ref, qsa)));
    }
    gsw.close();
  }
}
