// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

// TODO test that initialization bias is overcome!!!
class RSTQP_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        cliffwalk, qsa, ConstantLearningRate.one());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_rstqp.gif"), 100);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Infoline.print(cliffwalk, index, ref, qsa);
      TabularSteps.batch(cliffwalk, cliffwalk, rstqp);
      gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(cliffwalk, qsa, ref)));
    }
    gsw.close();
  }
}
