// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

// R1STQP algorithm is not suited for gambler's dilemma
/* package */ enum RSTQP_Gambler {
  ;
  public static void main(String[] args) throws Exception {
    GamblerModel gamblerModel = new GamblerModel(20, RationalScalar.of(4, 10));
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of(gamblerModel, qsa, //
        DefaultLearningRate.of(4, 0.71));
    ActionValueStatistics avs = new ActionValueStatistics(gamblerModel);
    try (AnimationWriter animationWriter1 = new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_rstqp.gif"), 100, TimeUnit.MILLISECONDS)) {
      try (AnimationWriter animationWriter2 = new GifAnimationWriter(HomeDirectory.Pictures("gambler_sac_rstqp.gif"), 200, TimeUnit.MILLISECONDS)) {
        int batches = 200;
        for (int index = 0; index < batches; ++index) {
          Infoline infoline = Infoline.print(gamblerModel, index, ref, qsa);
          TabularSteps.batch(gamblerModel, gamblerModel, rstqp, avs);
          animationWriter1.write(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
          animationWriter2.write(StateActionRasters.qsa( //
              gamblerRaster, DiscreteValueFunctions.rescaled(((DiscreteStateActionCounter) rstqp.sac()).inQsa(gamblerModel))));
          if (infoline.isLossfree())
            break;
        }
      }
    }
    // ---
    // ActionValueIteration avi = new ActionValueIteration(gambler, avs);
    // avi.setMachinePrecision();
    // avi.untilBelow(RealScalar.of(.0001));
    // Scalar error = DiscreteValueFunctions.distance(ref, avi.qsa());
    // System.out.println(error);
    // Export.of(UserHome.Pictures("gambler_avs.png"),
    // // GamblerHelper.qsaPolicyRef(gambler, avi.qsa(), ref)
    // StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref));
  }
}
