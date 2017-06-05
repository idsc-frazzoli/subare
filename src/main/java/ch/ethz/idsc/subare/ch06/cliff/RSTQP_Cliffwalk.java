// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

// R1STQP algorithm is not suited for gambler's dilemma
class RSTQP_Cliffwalk {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    Cliffwalk gambler = new Cliffwalk(12, 4);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        gambler, gambler, qsa);
    rstqp.setUpdateFactor(RealScalar.of(.1));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_rstqp.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + error.map(ROUND));
      rstqp.batch();
      rstqp.batch();
      gsw.append(ImageFormat.of(CliffwalkHelper.joinAll(gambler, qsa, ref)));
    }
    gsw.close();
  }
}
