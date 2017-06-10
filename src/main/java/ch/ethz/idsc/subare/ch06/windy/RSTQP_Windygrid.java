// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

// R1STQP algorithm is not suited for gambler's dilemma
class RSTQP_Windygrid {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        windygrid, windygrid, qsa);
    rstqp.setUpdateFactor(RealScalar.of(1));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/windygrid_qsa_rstqp.gif"), 250);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      System.out.println(index + " " + error.map(ROUND));
      rstqp.batch();
      gsw.append(ImageFormat.of(WindygridHelper.joinAll(windygrid, qsa, ref)));
    }
    gsw.close();
  }
}
