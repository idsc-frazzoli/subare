// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

// R1STQP algorithm is not suited for gambler's dilemma
class RSTQP_Windygrid {
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        windygrid, qsa, DefaultLearningRate.of(5, 1.0)); // TODO try learning rate
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("windygrid_qsa_rstqp.gif"), 250);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteValueFunctions.distance(qsa, ref);
      System.out.println(index + " " + error.map(Round._1));
      TabularSteps.batch(windygrid, windygrid, rstqp);
      gsw.append(ImageFormat.of(WindygridHelper.joinAll(windygrid, qsa, ref)));
    }
    gsw.close();
  }
}
