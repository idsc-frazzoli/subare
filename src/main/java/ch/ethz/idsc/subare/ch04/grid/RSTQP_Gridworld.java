// code by jph
package ch.ethz.idsc.subare.ch04.grid;

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
class RSTQP_Gridworld {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        gridworld, gridworld, qsa);
    rstqp.setUpdateFactor(RealScalar.of(.25));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_rstqp.gif"), 150);
    int EPISODES = 30;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + error.map(ROUND));
      rstqp.batch();
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa, ref)));
    }
    gsw.close();
  }
}
