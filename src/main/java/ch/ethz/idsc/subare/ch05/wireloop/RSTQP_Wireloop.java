// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

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

/** Example 4.1, p.82 */
class RSTQP_Wireloop {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        wireloop, wireloop, qsa);
    rstqp.setUpdateFactor(RealScalar.ONE);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_qsa_rstqp.gif"), 250);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + error.map(ROUND));
      rstqp.batch();
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, qsa)));
    }
    gsw.close();
  }
}
