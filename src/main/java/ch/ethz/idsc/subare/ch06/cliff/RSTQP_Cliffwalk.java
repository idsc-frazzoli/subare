// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

// TODO R1STQP algorithm is suited or not ??? 
class RSTQP_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        cliffwalk, qsa, DefaultLearningRate.of(5, 1.0)); // TODO try learning rate
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_rstqp.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      System.out.println(index + " " + error.map(Digits._1));
      TabularSteps.batch(cliffwalk, cliffwalk, rstqp);
      gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(cliffwalk, qsa, ref)));
    }
    gsw.close();
  }
}
