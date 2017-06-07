// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** {@link DequeDigest} for gridworld */
class DD_Gridworld {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  static void handle(SarsaType type, int n) throws Exception {
    System.out.println(type);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // used in egreedy
    Tensor learning = Subdivide.of(.3, .01, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_" + type + "" + n + ".gif"), 150);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar explore = epsilon.Get(index);
      Scalar alpha = learning.Get(index);
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + explore.map(ROUND) + " " + error.map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gridworld, qsa, explore);
      DequeDigest dequeDigest = new OriginalSarsa(gridworld, qsa, alpha, policyInterface);
      ExploringStartsBatch.apply(gridworld, dequeDigest, 4, policyInterface);
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa, ref)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 3);
    // handle(SarsaType.expected, 3);
    handle(SarsaType.qlearning, 3);
  }
}
