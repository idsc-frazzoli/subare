// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.sca.Round;

class RSTQP_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        cliffwalk, qsa, ConstantLearningRate.one());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_rstqp.gif"), 200);
    int batches = 20;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(cliffwalk, index, ref, qsa);
      TabularSteps.batch(cliffwalk, cliffwalk, rstqp);
      gsw.append(StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
    qsa.print(Round._2);
  }
}
