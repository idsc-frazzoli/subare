// code by jph
package ch.ethz.idsc.subare.demo.fish;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.sca.Round;

enum RSTQP_Fishfarm {
  ;
  public static void main(String[] args) throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster cliffwalkRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        fishfarm, qsa, ConstantLearningRate.one());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("fishfarm_qsa_rstqp.gif"), 200, TimeUnit.MILLISECONDS)) {
      int batches = 20;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(fishfarm, index, ref, qsa);
        TabularSteps.batch(fishfarm, fishfarm, rstqp);
        animationWriter.append(StateRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
    DiscreteUtils.print(qsa, Round._2);
  }
}
