// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** the R1STQP algorithm is cheating on the Windygrid
 * because TabularSteps starts in every state-action pair
 * instead of only the 1 start state of Windygrid */
class RSTQP_Windygrid {
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        windygrid, qsa, ConstantLearningRate.one());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("windygrid_qsa_rstqp.gif"), 250);
    int EPISODES = 40;
    for (int index = 0; index < EPISODES; ++index) {
      Infoline infoline = Infoline.print(windygrid, index, ref, qsa);
      TabularSteps.batch(windygrid, windygrid, rstqp);
      gsw.append(ImageFormat.of(WindygridHelper.joinAll(windygrid, qsa, ref)));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
  }
}
